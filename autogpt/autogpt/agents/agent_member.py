import logging
from enum import Enum
import uuid
import inspect
from g4f.Provider.not_working.Bestim import uuid4
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from autogpt.agent_factory.profile_generator import AgentProfileGenerator
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.agents.prompt_strategies.divide_and_conquer import (
    CommandRequest,
    DivideAndConquerAgentActionProposal,
    DivideAndConquerAgentPromptConfiguration,
    DivideAndConquerAgentPromptStrategy,
)
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.agent_protocol.api_router import TaskRequestBody
from forge.llm.providers import function_specs_from_commands
from forge.utils.exceptions import UnknownCommandError
from .task_management import TaskManagementComponent
from .agent_management import AgentManagementComponent
from forge.components.action_history import EpisodicActionHistory
from autogpt.app.config import AppConfig, ConfigBuilder
from forge.file_storage import FileStorage
from forge.llm.prompting import ChatPrompt
from forge.llm.providers.schema import AssistantChatMessage, ChatMessage, CompletionModelFunction
from forge.llm.providers import MultiProvider
from forge.models.action import ActionErrorResult, ActionResult, ActionSuccessResult
from forge.command import Command
from forge.agent_protocol.models.task import Task

logger = logging.getLogger(__name__)

class AgentTaskStatus(Enum):
    INITIAL = "INITIAL"
    DOING = "DOING"
    CHECKING = "CHECKING"
    REJECTED = "REJECTED"
    DONE = "DONE"

class AgentTaskSettings(BaseModel):
    input: str
    task_id: str
    parent_task_id: Optional[str]
    status: AgentTaskStatus
    sub_tasks: list[str]


class AgentMemberSettings(AgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: DivideAndConquerAgentPromptConfiguration = Field(
        default_factory=(
            lambda: DivideAndConquerAgentPromptStrategy.default_configuration.copy(
                deep=True
            )
        )
    )
    role: str
    prompt: str
    boss_id: Optional[str]
    recruiter_id: Optional[str]
    create_agent: bool
    members: list[str]
    tasks: list[AgentTaskSettings]
    history: EpisodicActionHistory[DivideAndConquerAgentActionProposal] = Field(
        default_factory=EpisodicActionHistory[DivideAndConquerAgentActionProposal]
    )


class ProposeActionResult:
    commands: list[CommandRequest]
    agent: "AgentMember"

    def __init__(self, commands: list[CommandRequest], agent: "AgentMember") -> None:
        self.commands = commands
        self.agent = agent


class CommandActionResult:
    action_result: ActionResult
    command: str

    def __init__(self, action_result: ActionResult, command: str) -> None:
        self.action_result = action_result
        self.command = command


class AgentMember(Agent):
    boss: Optional["AgentMember"]
    recruiter: Optional["AgentMember"]
    # tasks: list["AgentTask"]
    members: list["AgentMember"]
    group: "AgentGroup"

    def recursive_assign_group(self, group: "AgentGroup"):
        self.group = group
        for members in self.members:
            members.recursive_assign_group(group)

    def get_list_of_all_your_team_members(self) -> list["AgentMember"]:
        members = []
        print("self: " + self.state.agent_id)
        for member in self.members:
            print("member: " + member.state.agent_id)
            members.extend(member.get_list_of_all_your_team_members())
        members.append(self)
        return members

    def print_state(self):
        logger.info(f"Agent ID: {self.state.agent_id}")
        logger.info(f"AI Name: {self.state.ai_profile.ai_name}")
        logger.info(f"AI Role: {self.state.ai_profile.ai_role}")
        logger.info("Tasks:")
        for task in self.state.tasks:
            logger.info(f"  {task.input}")
        logger.info("Members:")
        for member in self.members:
            member.print_state()

    def __init__(
        self,
        settings: AgentMemberSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        legacy_config: AppConfig,
        boss: Optional["AgentMember"] = None,
        recruiter: Optional["AgentMember"] = None,
        # tasks: list["AgentTask"] = [],
        members: list["AgentMember"] = [],
    ):
        super().__init__(settings, llm_provider, file_storage, legacy_config)

        self.boss = boss
        self.recruiter = recruiter
        # self.tasks = tasks
        self.members = members
        self.prompt_strategy = DivideAndConquerAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )
        self.task_management_component = TaskManagementComponent(self)
        self.agent_management_component = AgentManagementComponent(self)

    async def build_prompt(
        self,
        tasks: list["AgentTask"],
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """
        # Get directives
        resources = await self.run_pipeline(DirectiveProvider.get_resources)
        constraints = await self.run_pipeline(DirectiveProvider.get_constraints)
        best_practices = await self.run_pipeline(DirectiveProvider.get_best_practices)

        directives = self.state.directives.copy(deep=True)
        directives.resources += resources
        directives.constraints += constraints
        directives.best_practices += best_practices

        self.commands = await self.run_pipeline(CommandProvider.get_commands)

        prompt = self.prompt_strategy.build_prompt(
            include_os_info=True,
            tasks=tasks,
            agent_member=self,
            ai_profile=self.state.ai_profile,
            ai_directives=directives,
            commands=function_specs_from_commands(self.commands),
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt

    async def execute_commands(
        self, commands: list[CommandRequest]
    ) -> list[CommandActionResult]:
        results = []

        for command in commands:
            # self.event_history.register_action(
            #     Action(
            #         name=command.command,
            #         args=command.args,
            #         reasoning="",
            #     )
            # )

            command_template = self._get_command(command.command)
            try:
                result = command_template(**command.args)
                if inspect.isawaitable(result):
                    result = await result
                results.append(
                    CommandActionResult(action_result=ActionSuccessResult(outputs=result), command=command.command)
                )
            except Exception as e:
                results.append(
                    CommandActionResult(action_result=ActionErrorResult(reason=str(e)), command=command.command)
                )
        return results


    def _get_command(self, command_name: str) -> Command:
        for command in reversed(self.commands):
            if command_name in command.names:
                return command

        raise UnknownCommandError(
            f"Cannot execute command '{command_name}': unknown command."
        )

    async def create_task(self, task_request: TaskRequestBody, parent_task_id:str|None=None):
        # task = AgentTask(
        #     input=task_request.input,
        #     additional_input=task_request.additional_input,
        #     status=AgentTaskStatus.INITIAL.value,
        #     created_at=datetime.now(),
        #     modified_at=datetime.now(),
        #     task_id=str(uuid.uuid4()),
        #     sub_tasks=[],
        #     artifacts=[],
        #     parent_task_id=None,
        #     parent_task=None
        # )
        agentTaskSetting = AgentTaskSettings(
            input=task_request.input,
            parent_task_id=parent_task_id,
            task_id=str(uuid.uuid4()),
            status=AgentTaskStatus.INITIAL.value,
            sub_tasks=[]
        )
        # self.tasks.append(task)
        self.state.tasks.append(agentTaskSetting)
        await self.file_manager.save_state()

    async def recursive_propose_action(self) -> list[ProposeActionResult]:
        result = [
            ProposeActionResult(agent=self, commands=await self.single_propose_action())
        ]
        for agent_member in self.members:
            result = result + (await agent_member.recursive_propose_action())
        return result

    async def single_propose_action(self) -> list[CommandRequest]:
        current_tasks = []
        for task in self.state.tasks:
            if task.status == AgentTaskStatus.REJECTED:
                task.status = AgentTaskStatus.INITIAL

            elif task.status == AgentTaskStatus.DOING:
                # sub_tasks_done = all(sub_task.status == AgentTaskStatus.DONE for sub_task in task.sub_tasks)
                # # sub_tasks_checking = any(sub_task.status == AgentTaskStatus.CHECKING for sub_task in task.sub_tasks)

                # if sub_tasks_done:
                #     task.status = AgentTaskStatus.CHECKING
                # elif sub_tasks_checking:
                current_tasks.append(task)

            elif task.status == AgentTaskStatus.INITIAL:
                current_tasks.append(task)
                task.status = AgentTaskStatus.DOING

        commands: list[CommandRequest] = []
        if current_tasks:
            logger.info(f"tasks: {str(current_tasks)}")
            prompt = await self.build_prompt(tasks=current_tasks)
            result = await self.llm_provider.create_chat_completion(
                prompt.messages,
                model_name=self.config.smart_llm,
                functions=prompt.functions,
                completion_parser=lambda r: self.parse_and_process_response(r),
            )
            print(commands)
            commands = result.parsed_result
        await self.file_manager.save_state()
        return commands

    def parse_and_process_response(
        self, llm_response: AssistantChatMessage
    ) -> list[CommandRequest]:
        result = self.prompt_strategy.parse_response_content(llm_response)
        return result



# class AgentTask(Task):
#     parent_task_id: Optional[str]
#     status: AgentTaskStatus
#     parent_task: Optional["AgentTask"]
#     sub_tasks: list["AgentTask"]


async def create_agent_member(
    state: AgentMemberSettings,
    app_config: AppConfig,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
    boss: Optional["AgentMember"] = None,
    recruiter: Optional["AgentMember"] = None,
) -> "AgentMember":
    agent_member = AgentMember(
        settings=state,
        boss=boss,
        recruiter=recruiter,
        llm_provider=llm_provider,
        file_storage=file_storage,
        legacy_config=app_config,
    )

    if boss:
        boss.members.append(agent_member)
        boss.group.reload_members()
        boss.state.members.append(agent_member.state.agent_id)
        await boss.file_manager.save_state()

    await agent_member.file_manager.save_state()
    return agent_member


async def generate_agent_settings_for_task(
    role: str,
    prompt: str,
    boss_id: Optional[str],
    recruiter_id: Optional[str],
    tasks: list[AgentTaskSettings],
    members: list[str],
    create_agent: bool,
    llm_provider: MultiProvider,
    app_config,
) -> AgentMemberSettings:
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    profile_prompt = agent_profile_generator.build_prompt(prompt)
    output = (
        await llm_provider.create_chat_completion(
            profile_prompt.messages,
            model_name=app_config.smart_llm,
            functions=profile_prompt.functions,
        )
    ).response

    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    return AgentMemberSettings(
        agent_id=str(uuid.uuid4()),
        role=role,
        prompt=prompt,
        boss_id=boss_id,
        recruiter_id=recruiter_id,
        tasks=tasks,
        members=members,
        create_agent=create_agent,
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        directives=ai_directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
        ),
        history=[],
    )


async def create_agent_member_from_task(
    role: str,
    prompt: str,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
    boss_id=None,
):
    config = ConfigBuilder.build_config_from_env()
    config.logging.plain_console_output = True
    config.continuous_mode = False
    config.continuous_limit = 20
    config.noninteractive_mode = True
    settings = await generate_agent_settings_for_task(
        role=role,
        prompt=prompt,
        boss_id=boss_id,
        recruiter_id=None,
        tasks=[],
        members=[],
        create_agent=True,
        app_config=config,
        llm_provider=llm_provider,
    )
    agent_member = await create_agent_member(
        file_storage=file_storage,
        app_config=config,
        state=settings,
        llm_provider=llm_provider,
    )
    return agent_member
