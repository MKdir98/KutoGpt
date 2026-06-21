"""Commands to search the web with"""

from __future__ import annotations
from typing import Iterator

from forge.file_storage import FileStorageBackendName, get_storage
from autogpt.app.config import ConfigBuilder
from forge.file_storage import get_storage
from forge.models.json_schema import JSONSchema
from forge.agent.protocols import CommandProvider
from forge.command import Command, command
import traceback

class AgentManagementComponent(CommandProvider):
    """Component for manage agents."""

    def __init__(self, agent: 'AgentMember') -> None:
        self.agent = agent

    def get_commands(self) -> Iterator[Command]:
        if self.agent.state.create_agent:
            yield self.create_agent
        else:
            yield self.request_agent

    @command(
        ["create_agent"],
        "Create a new agent member for someone. The prompt for this step should be create someone to do this task.",
        {
            "prompt": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description for agent that one to be created",
                required=True,
            ),
            "role": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="role of agent member that one be created",
                required=True,
            ),
            "boss_id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The agent who will be boss of new agent id",
                required=True,
            ),
        },
    )
    async def create_agent(self, prompt: str, role: str, boss_id: str) -> str:
        """Create new agent for some one

        Args:
            prompt (str): The description for agent that one to be created.
            role (str): role of agent member that one be created.

        """
        from autogpt.agents.agent_member import generate_agent_settings_for_task, create_agent_member
        try:
            config = ConfigBuilder.build_config_from_env()
            config.logging.plain_console_output = True
            config.continuous_mode = False
            config.continuous_limit = 20
            config.noninteractive_mode = True

            settings = await generate_agent_settings_for_task(
                role=role,
                boss_id=boss_id,
                recruiter_id=None,
                tasks=[],
                members=[],
                create_agent=False,
                prompt=prompt,
                app_config=config,
                llm_provider=self.agent.llm_provider,
            )
            group = self.agent.group
            boss = group.members[boss_id]
            config = ConfigBuilder.build_config_from_env()
            local = config.file_storage_backend == FileStorageBackendName.LOCAL
            restrict_to_root = not local or config.restrict_to_workspace
            file_storage = get_storage(
                config.file_storage_backend,
                root_path="data_group",
                restrict_to_root=restrict_to_root,
            )
            file_storage.initialize()

            await create_agent_member(
                file_storage=file_storage,
                app_config=config,
                state=settings,
                boss=boss,
                llm_provider=self.agent.llm_provider,
            )
            return f"{role} created"
        except Exception as ex:
            traceback.print_exc()
            return f"can't create {role}"

    @command(
        ["request_agent"],
        "Request a new agent member for someone. The prompt for this step should be create someone to do this task.",
        {
            "prompt": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description for agent that one to be created",
                required=True,
            ),
            "role": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Role of agent member that one be created",
                required=True,
            ),
            "boss_id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The agent id that is going to be the boss of new agent",
                required=True,
            ),
        },
    )
    async def request_agent(self, prompt: str, role: str, boss_id: str) -> str:
        """Request new agent for some one

        Args:
            prompt (str): The description for agent that one to be created.
            role (str): role of agent member that one be created.
            boss_id (str): The agent id that is going to be the boss of new agent.

        """
        try:
            if self.agent.recruiter != None:
                await self.agent.recruiter.create_task(
                    task_request=TaskRequestBody(
                        input=f"hire someone with {role} and this prompt: {prompt} for agent with id {boss_id}"
                    )
                )
                return f"create task for recruiter to hire {role}"
            elif self.agent.boss != None:
                await self.agent.boss.create_task(
                    task_request=TaskRequestBody(
                        input=f"hire someone with {role} and this prompt: {prompt} for agent with id {boss_id}"
                    )
                )
                return f"create task for boss to hire {role}"
            else:
                raise Exception("We can't hire someone ")
        except Exception as ex:
            print(ex)
            return f"can't create {role}"
