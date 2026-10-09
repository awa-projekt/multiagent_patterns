"""Pattern 10 - Skills (progressive disclosure), built with `create_skills_agent` + `Skill`."""

from agentpatterns import Skill, create_skills_agent
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.native.skills import SKILLS
from email_assistant.schemas import EmailResolution
from email_assistant.tools import DOMAIN_TOOLS, lookup_customer


def build_graph(model):
    agent = create_skills_agent(
        model,
        [Skill(name, s["description"], s["instructions"], tools=DOMAIN_TOOLS[name]) for name, s in SKILLS.items()],
        system_prompt=prompts.SKILLS_AGENT,
        tools=[lookup_customer],
        response_format=EmailResolution,
        name="service_generalist",
    )
    return build_email_workflow(agent, "skills")
