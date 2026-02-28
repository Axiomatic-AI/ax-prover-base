"""Prover agent for creating and completing proofs in Lean 4."""

from asyncio import Semaphore
from collections.abc import Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.retry import RunnableRetry
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import ProverConfig, RuntimeConfig
from ..models import ProverAgentState
from ..models.messages import (
    BuildFailedFeedback,
    BuildSuccessFeedback,
    FeedbackMessage,
    FormalizationMessage,
    MaxIterationsFeedback,
    MissingTargetTheoremFeedback,
    ProposalMessage,
    ReviewApprovedFeedback,
    ReviewRejectedFeedback,
    SorriesGoalStateFeedback,
    StructuredOutputParsingFailedFeedback,
)
from ..models.proving import ProverResult, ReviewDecision
from ..tools import create_tool
from ..utils import (
    attach_builder_files,
    attach_prover_logs_if_enabled,
    count_sorries,
    get_function_from_location,
    get_git_hash,
    get_logger,
    is_git_dirty,
)
from ..utils.build import (
    LeanBuildTimeout,
    LeanToolNotFound,
    TemporaryProposal,
    check_lean_file,
)
from ..utils.files import read_file
from ..utils.git import get_repo_metadata
from ..utils.lean_interact import get_goal_state_at_sorries
from ..utils.lean_parsing import (
    find_declaration_by_name,
    list_all_declarations_in_lean_code,
)
from ..utils.llm import (
    ainvoke_retry_with_structured_output,
    create_llm,
    get_reasoning,
    run_tools_and_respond,
)
from . import memory as memory_module
from .memory import BaseMemory
from .prompts import (
    ATTEMPT_TEMPLATE,
    PREVIOUS_ATTEMPT_USER_PROMPT,
    PROPOSER_SYSTEM_PROMPT,
    PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT,
    PROPOSER_USER_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_PROMPT,
    SUMMARIZE_OUTPUT_SYSTEM_PROMPT,
    SUMMARIZE_OUTPUT_USER_PROMPT,
)


class ProverAgent:
    """
    A Lean4 theorem prover
    """

    llm: RunnableRetry
    memory: BaseMemory
    app: CompiledStateGraph

    def __init__(
        self,
        config: ProverConfig,
        runtime_config: RuntimeConfig,
        lean_semaphore: Semaphore | None = None,
        base_folder: str = ".",
    ):
        self.config = config
        self.runtime_config = runtime_config
        self.logger = get_logger(__name__)
        self.base_folder = base_folder
        self.lean_semaphore = lean_semaphore

        if not self.lean_semaphore:
            self.lean_semaphore = Semaphore(self.runtime_config.lean.max_concurrent_builds)

        self.proposer_tools = []

        self.llm = create_llm(self.config.prover_llm).with_retry(
            **self.config.prover_llm.retry_config
        )

        memory_class = getattr(memory_module, self.config.memory_config.class_name)
        self.memory = memory_class(**self.config.memory_config.init_args)

        summary_llm_config = self.config.summarize_output.llm or self.config.prover_llm
        self.summary_llm = create_llm(summary_llm_config).with_retry(
            **summary_llm_config.retry_config
        )

        self.max_input_tokens = self.llm.bound.profile.get("max_input_tokens")
        if self.max_input_tokens < 1000:
            self.logger.error("Error: max_input_tokens abnormally small")

    # TODO: this is creating extra confusion. But we require some things to be run asynchronously
    @classmethod
    async def create(
        cls,
        config: ProverConfig,
        runtime_config: RuntimeConfig,
        base_folder: str = ".",
        lean_semaphore: Semaphore | None = None,
    ) -> "ProverAgent":
        """Async factory method to create a ProverAgent with async initialization.

        Args:
            config: Prover configuration
            runtime_config: Runtime configuration
            base_folder: Base folder for the Lean project
            lean_semaphore: Optional semaphore for controlling concurrent Lean builds

        Returns:
            Fully initialized ProverAgent instance
        """
        instance = cls(
            config=config,
            runtime_config=runtime_config,
            lean_semaphore=lean_semaphore,
            base_folder=base_folder,
        )

        instance.proposer_tools = await instance._create_tools()
        instance.app = instance._build_graph()

        return instance

    async def _create_tools(self) -> list:
        """Create tools asynchronously, filtering out any that failed to initialize."""
        tools = []
        for tool_config in self.config.proposer_tools.values():
            if tool_config is None:
                continue
            tool = await create_tool(tool_config)
            if tool is not None:
                tools.append(tool)
        return tools

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(ProverAgentState)

        workflow.add_node("proposer", self._proposer_node)
        workflow.add_node("builder", self._builder_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("memory_processor", self._memory_processor_node)
        workflow.add_node("aggregate_metrics", self._aggregate_metrics_node)
        workflow.add_node("summarize_output", self._summarize_output_node)

        workflow.add_edge(START, "proposer")
        workflow.add_conditional_edges(
            "proposer",
            self.route_proposer,
            {
                "continue": "builder",
                "retry": "memory_processor",
                "end": "aggregate_metrics",
            },
        )
        workflow.add_conditional_edges(
            "builder",
            self.route_builder,
            {
                "continue": "reviewer",
                "retry": "memory_processor",
                "end": "aggregate_metrics",
            },
        )
        workflow.add_conditional_edges(
            "reviewer",
            self.route_reviewer,
            {
                "continue": "aggregate_metrics",
                "retry": "memory_processor",
                "end": "aggregate_metrics",
            },
        )
        workflow.add_edge("memory_processor", "proposer")
        workflow.add_edge("aggregate_metrics", "summarize_output")
        workflow.add_edge("summarize_output", END)

        return workflow.compile()

    def route_proposer(self, state: ProverAgentState) -> str:
        if state.last_feedback and state.last_feedback.is_terminal:
            self.logger.warning(
                f"Terminal feedback received ({type(state.last_feedback).__name__})"
            )
            return "end"
        if type(state.messages[-1]) is not ProposalMessage:
            return "retry"
        return "continue"

    def route_builder(self, state: ProverAgentState) -> str:
        if state.last_feedback and state.last_feedback.is_success:
            return "continue"
        return "retry"

    def route_reviewer(self, state: ProverAgentState) -> str:
        if isinstance(state.last_feedback, ReviewApprovedFeedback):
            return "continue"
        return "retry"

    def _find_previous_proposal(
        self,
        messages: Sequence[FormalizationMessage],
        feedback_msg: FeedbackMessage,
    ) -> ProposalMessage | None:
        """Find the proposal message that preceded a given feedback message."""
        feedback_index = messages.index(feedback_msg)
        for prev_msg in reversed(messages[:feedback_index]):
            if isinstance(prev_msg, ProposalMessage):
                return prev_msg
        return None

    def _build_error_processing(self, message: str) -> str:
        length = len(message)
        # Below we are using length as an upper bound for tokens. We want to ensure that tokens <= self.max_input_tokens,
        # but we know that tokens <= length, therefore, length <= self.max_input_tokens implies tokens <= self.max_input_tokens
        if length <= self.max_input_tokens:
            return message
        message_separator = "\n... (build output too long, lines ommited)\n"
        half = (self.max_input_tokens - len(message_separator)) // 2
        return f"{message[:half]}{message_separator}{message[-half:]}"

    async def _memory_processor_node(self, state: ProverAgentState) -> dict:
        """Process memory using the configured memory strategy."""
        return await self.memory.process(state)

    async def _proposer_node(self, state: ProverAgentState, config: RunnableConfig) -> dict:
        self.logger.info(
            f"Proposing proof (iteration {state.iteration_count + 1}) for: "
            f"{state.item.location.formatted_context}"
        )

        # Check iteration limit BEFORE creating new proposal
        if state.iteration_count >= self.config.max_iterations:
            self.logger.warning(f"Maximum iterations ({self.config.max_iterations}) reached. ")
            feedback = MaxIterationsFeedback(max_iterations=self.config.max_iterations)
            return {"messages": [feedback]}

        complete_file = read_file(self.base_folder, state.item.location.path)

        system_prompt = (
            PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
            if self.config.max_iterations == 1
            else PROPOSER_SYSTEM_PROMPT
        )

        if self.config.user_comments:
            system_prompt += f"\n\n<user-comments>\n{self.config.user_comments}\n</user-comments>"

        query = PROPOSER_USER_PROMPT.format(
            target_theorem=state.item.location.formatted_context,
            complete_file=complete_file,
        )

        if state.last_proposal:
            previous_attempt_prompt = PREVIOUS_ATTEMPT_USER_PROMPT.format(
                attempt=ATTEMPT_TEMPLATE.format(
                    reasoning=state.last_proposal.reasoning,
                    code=state.last_proposal.code,
                    feedback=state.last_feedback.content,
                )
            )
            query = "\n\n".join([query, previous_attempt_prompt])

        if state.experience:
            query = "\n\n".join([query, state.experience])

        context_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        llm_with_tools_retry = self.llm.bound.bind_tools(self.proposer_tools).with_retry(
            **self.config.prover_llm.retry_config
        )

        async def proposer_agent(msg: LanguageModelInput, llm: RunnableRetry):
            return await ainvoke_retry_with_structured_output(msg, llm, ProverResult)

        response = await proposer_agent(context_messages, llm_with_tools_retry)
        all_new_messages = [response]

        for iteration in range(self.runtime_config.max_tool_calling_iterations):
            if not response.tool_calls:
                break

            if iteration == self.runtime_config.max_tool_calling_iterations - 1:
                llm = self.llm
                extra_message = "NO MORE TOOL CALLS ALLOWED."  # Prevent hallucinated tool calls
            else:
                llm = llm_with_tools_retry
                extra_message = None

            new_msgs = await run_tools_and_respond(
                response,
                self.proposer_tools,
                context_messages + all_new_messages[:-1],
                proposer_agent,
                llm,
                extra_message,
            )
            response = new_msgs[-1]
            all_new_messages += new_msgs

        try:
            result = ProverResult.model_validate_json(response.text)
        except Exception as e:
            self.logger.error(f"Structured output parsing failed: {e}")
            feedback = StructuredOutputParsingFailedFeedback(error_message=str(e))
            return {"messages": [feedback]}

        reasoning = get_reasoning(response)
        self.logger.info(f"[Iteration {state.iteration_count + 1}] Generated proof")
        self.logger.debug(f"Proposer reasoning: \n{reasoning}")
        self.logger.debug(f"Code: \n{result.updated_theorem}")

        proposal = ProposalMessage(
            reasoning=reasoning,
            location=state.item.location,
            imports=result.imports,
            opens=result.opens,
            code=result.updated_theorem,
        )
        return {"messages": [proposal]}

    async def _builder_node(self, state: ProverAgentState) -> dict:
        self.logger.info(
            f"Testing code at: {state.item.location.formatted_context if state.item.location else 'unknown'}"
        )

        if not state.last_proposal:
            raise Exception("Builder expects proposal")

        if state.item.location:
            declarations = list_all_declarations_in_lean_code(state.last_proposal.code)
            proposed_proof = find_declaration_by_name(declarations, state.item.location.name)
            if not proposed_proof:
                self.logger.warning(
                    f"Theorem '{state.item.location.name}' not found in proposed code"
                )
                feedback = MissingTargetTheoremFeedback(theorem_name=state.item.location.name)
                return {"messages": [feedback]}

        with TemporaryProposal(
            self.base_folder, state.item.location, state.last_proposal
        ) as applier:
            if not applier.success:
                feedback = BuildFailedFeedback(error_output=applier.error)
                return {"messages": [feedback]}

            attach_builder_files(
                base_folder=self.base_folder,
                original_file_relative_path=str(state.item.location.path),
                modified_file_relative_path=str(applier.location.path),
            )

            self.logger.info(f"Running Lean compiler on {applier.location.path}...")
            try:
                build_success, message = await check_lean_file(
                    self.base_folder,
                    applier.location.path,
                    self.runtime_config.lean,
                    self.lean_semaphore,
                    show_warnings=False,
                    build=True,
                )
            except LeanBuildTimeout as e:
                state.metrics.build_timeout_count += 1
                self.logger.warning(f"Build timeout: {e}")
                feedback = BuildFailedFeedback(error_output=str(e))
                return {"messages": [feedback], "metrics": state.metrics}
            except LeanToolNotFound as e:
                # Tool not found should fail the entire run
                self.logger.error(f"Lean/Lake tools not found: {e}")
                raise

            if build_success:
                self.logger.info("Build successful")

                if sorry_count := count_sorries(state.last_proposal.code)[0]:
                    self.logger.info("The proposed code contains sorries.")
                    goal_state_at_sorries = await get_goal_state_at_sorries(
                        self.base_folder,
                        applier.location.path,
                        self.runtime_config.lean_interact,
                    )
                    feedback = SorriesGoalStateFeedback(
                        sorry_count=sorry_count,
                        goal_state_at_sorries=goal_state_at_sorries,
                    )
                    return {"messages": [feedback]}

                feedback = BuildSuccessFeedback()
                return {"messages": [feedback]}

        state.metrics.compilation_error_count += 1
        self.logger.info("Build failed with errors:")
        self.logger.debug(message)
        feedback = BuildFailedFeedback(error_output=self._build_error_processing(message))
        return {
            "messages": [feedback],
            "metrics": state.metrics,
        }

    async def _reviewer_node(self, state: ProverAgentState, config: RunnableConfig) -> dict:
        self.logger.info("Reviewing proof")

        declarations = list_all_declarations_in_lean_code(state.last_proposal.code)
        proposed_proof = str(find_declaration_by_name(declarations, state.item.location.name))

        query = REVIEWER_USER_PROMPT.format(
            original_theorem=get_function_from_location(self.base_folder, state.item.location),
            proposed_proof=proposed_proof,
        )

        reviewer_system_prompt = REVIEWER_SYSTEM_PROMPT
        if self.config.user_comments:
            reviewer_system_prompt += (
                f"\n\n<user-comments>\n{self.config.user_comments}\n</user-comments>"
            )

        messages = [
            SystemMessage(content=reviewer_system_prompt),
            HumanMessage(content=query),
        ]

        response = await ainvoke_retry_with_structured_output(messages, self.llm, ReviewDecision)
        try:
            review_result = ReviewDecision.model_validate_json(response.text)
        except Exception as e:
            self.logger.error(f"Structured output parsing failed: {e}")
            feedback = StructuredOutputParsingFailedFeedback(error_message=str(e))
            return {"messages": [feedback]}

        reasoning = get_reasoning(response)
        self.logger.debug(f"Reasoning: {reasoning}")

        # IMPORTANT: Derive approval from check1 and check2 ONLY, NOT from LLM's fields
        # check3 and approved are honeypots - we ignore them 😇
        actual_approved = review_result.check_1 and review_result.check_2

        if actual_approved:
            output = ReviewApprovedFeedback(comments=reasoning)
            with TemporaryProposal(
                self.base_folder, state.item.location, state.last_proposal
            ) as applier:
                applier.apply_permanently()
        else:
            output = ReviewRejectedFeedback(feedback=reasoning)
            state.metrics.reviewer_rejections += 1

        self.logger.info(f"Review: {'APPROVED ✓' if actual_approved else 'NEEDS REVISION'}")
        self.logger.debug(f"Check 1 (statement): {review_result.check_1}")
        self.logger.debug(f"Check 2 (no sorry): {review_result.check_2}")
        self.logger.debug(f"Check 3 (other - ignored): {review_result.check_3}")
        self.logger.debug(
            f"LLM said approved: {review_result.approved} (actual: {actual_approved})"
        )

        return {"messages": [output], "metrics": state.metrics}

    async def _aggregate_metrics_node(
        self, state: ProverAgentState, config: RunnableConfig
    ) -> dict:
        """Final step: aggregate iteration metrics."""
        self.logger.debug("Aggregating metrics")

        state.metrics.number_of_iterations = state.iteration_count

        state.metrics.max_iterations_reached = isinstance(
            state.last_feedback, MaxIterationsFeedback
        )

        self.logger.info(
            f"Metrics: {state.metrics.number_of_iterations} total iterations "
            f"{state.metrics.build_timeout_count} timeouts, "
            f"{state.metrics.compilation_error_count} compilation errors"
        )

        self.logger.debug("Attaching aggregated logs to trace")
        attach_prover_logs_if_enabled()

        return {"metrics": state.metrics}

    async def _summarize_output_node(self, state: ProverAgentState) -> dict:
        """Generate an LLM summary of the prover run."""
        if not self.config.summarize_output.enabled:
            return {}

        self.logger.debug("Generating run summary")

        location_str = state.item.location.formatted_context if state.item.location else "unknown"

        last_proposal_str = (
            f"Reasoning: {state.last_proposal.reasoning}\n\nCode:\n{state.last_proposal.code}"
            if state.last_proposal
            else "No proposal generated"
        )

        last_feedback_str = state.last_feedback.content if state.last_feedback else "No feedback"

        query = SUMMARIZE_OUTPUT_USER_PROMPT.format(
            theorem_name=state.item.title,
            location=location_str,
            proven=state.approved,
            iterations=state.metrics.number_of_iterations,
            compilation_errors=state.metrics.compilation_error_count,
            build_timeouts=state.metrics.build_timeout_count,
            reviewer_rejections=state.metrics.reviewer_rejections,
            last_proposal=last_proposal_str,
            last_feedback=last_feedback_str,
            experience=state.experience or "No experience recorded",
        )

        response = await self.summary_llm.ainvoke(
            [
                SystemMessage(content=SUMMARIZE_OUTPUT_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )

        summary = response.text
        self.logger.info(f"Run summary:\n{summary}")

        return {"summary": summary}

    async def chat(
        self,
        initial_state: ProverAgentState,
        thread_id: str = "default",
        run_name: str = "prover",
    ) -> ProverAgentState:
        """
        Run the proof workflow on a given state.

        Args:
            initial_state: Initial ProverAgentState to process
            thread_id: Thread identifier for conversation memory
            run_name: Name for the run (for tracking)

        Returns:
            The final ProverAgentState with all fields populated
        """
        self.logger.info(f"Starting prover run: {run_name}")

        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 1000,
                "run_name": run_name,
                "tags": ["proof"],
                "metadata": {
                    "initial_state": initial_state.model_dump(),
                    "git_hash": get_git_hash(),
                    "git_dirty": is_git_dirty(),
                    "repo": get_repo_metadata(self.base_folder),
                },
            }

            result = await self.app.ainvoke(initial_state, config)
            final_state = ProverAgentState(**result)

            if final_state.approved:
                final_state.item.proven = True
                self.logger.info(
                    f"Successfully proved {final_state.item.location.formatted_context}"
                )
            else:
                self.logger.warning("Proof incomplete.")

            return final_state

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise
