# pickleball_scheduler/host_agent.py

import asyncio
from typing import Dict, List
from google.agent_development_kit import Agent, Tool

# Import our external tools
from tools import list_courts_availability, book_court

class PickleballHostAgent:
    """
    The central host agent responsible for coordinating with friends
    and booking pickleball courts.
    """
    def __init__(self, remote_agent_urls: List[str]):
        """
        Initializes the host agent and prepares connections to remote friend agents.

        Args:
            remote_agent_urls (List[str]): A list of base URLs for the remote friend agents.
        """
        self.remote_urls = remote_agent_urls
        self.agents: Dict[str, Dict] = {} # To store info about remote agents
        
        # In a real app, this would hold active A2A client objects
        # self.a2a_clients: Dict[str, A2AClient] = {} 
        print("Host Agent Initializing...")
        
        # Step 1: Prepare agent connections by fetching their "Agent Cards"
        self._prepare_remote_agents()

        # Step 2: Create the underlying Google ADK Agent
        self.adk_agent = self._create_adk_agent()
        print("\nHost Agent Ready. You can now start the conversation.")

    def _prepare_remote_agents(self):
        """
        For each URL, gets the agent card and saves agent information.
        This is a MOCK process. In a real A2A implementation, this would
        involve an HTTP request to an endpoint on the remote agent server.
        """
        print("Preparing connections to remote agents...")
        for url in self.remote_urls:
            # TODO: Replace this mock logic with a real A2A handshake.
            # e.g., response = requests.get(f"{url}/.well-known/agent-card")
            # agent_info = response.json()
            if "8001" in url: # LangGraph Agent
                agent_info = {
                    "name": "Alice_LangGraph",
                    "description": "An agent to check Alice's availability. Responds with her free time slots.",
                    "url": url,
                }
            elif "8002" in url: # CrewAI Agent
                agent_info = {
                    "name": "Bob_CrewAI",
                    "description": "An agent to check Bob's availability. Knows his preferred days.",
                    "url": url,
                }
            elif "8003" in url: # Google ADK Agent
                agent_info = {
                    "name": "Charlie_ADK",
                    "description": "An agent for Charlie's schedule. Can confirm or deny invitations.",
                    "url": url,
                }
            else:
                continue

            agent_name = agent_info["name"]
            self.agents[agent_name] = agent_info
            print(f"  - Registered remote agent: {agent_name} at {url}")
            
            # TODO: Initialize and store a real A2A client for this agent
            # self.a2a_clients[agent_name] = A2AClient(agent_info)


    def _build_system_prompt(self) -> str:
        """
        Constructs the root instructions for the agent, dynamically including
        the list of available agents to talk to.
        """
        core_instructions = """
You are a master pickleball scheduler. Your goal is to find a time that works for a group of friends, check court availability, and book a court.

To do this, you must first communicate with each friend's personal AI agent to determine their availability. Use the `send_message_to_friend` tool for this. You can ask them open-ended questions like "Are you free on Friday evening?".

Once you have availabilities, use the `list_courts_availability` tool.

Finally, after confirming a time with everyone, use the `book_court` tool to make the reservation. Be sure to include all player names in the booking.
"""

        agents_available_prompt = "\nAGENTS AVAILABLE:\n"
        if not self.agents:
            agents_available_prompt += "No remote agents are currently connected.\n"
        else:
            for name, info in self.agents.items():
                agents_available_prompt += f"- **{name}**: {info['description']}\n"
        
        return core_instructions + agents_available_prompt

    def send_message_to_friend(self, friend_name: str, message: str) -> str:
        """
        Sends a message to a specific friend's remote agent using the A2A protocol.

        Args:
            friend_name (str): The name of the friend's agent (e.g., 'Alice_LangGraph').
            message (str): The message to send.

        Returns:
            str: The response from the remote agent.
        """
        print(f"[*] Host Agent: Attempting to send message to '{friend_name}': '{message}'")
        
        # 1. Make sure we know this agent
        if friend_name not in self.agents:
            return f"Error: No agent found with the name '{friend_name}'. Available agents are: {list(self.agents.keys())}"

        # 2. Make sure we are connected (or connect now)
        # TODO: Implement actual A2A connection logic here.
        # client = self.a2a_clients[friend_name]
        # if not client.is_connected():
        #     client.connect()
        print(f"  - Connection to {friend_name} established (simulated).")
        
        # 3. Send the message using A2A and get a response
        # TODO: Replace this mock logic with a real A2A message exchange.
        # response = await client.send_message(message)
        # return response
        
        # Mock responses for demonstration purposes
        if "friday" in message.lower() and "alice" in friend_name.lower():
            return "Response from Alice_LangGraph: 'Friday sounds great! I'm free after 5 PM.'"
        elif "bob" in friend_name.lower():
             return "Response from Bob_CrewAI: 'I can do Friday at 6 PM or any time Saturday.'"
        else:
            return f"Response from {friend_name}: 'Acknowledged. I will check my schedule for: {message}'"


    def _create_adk_agent(self) -> Agent:
        """
        Creates and configures the Google ADK Agent instance.
        """
        print("\nBuilding ADK Agent...")
        system_prompt = self._build_system_prompt()

        # Pass in the agent's own method as a tool!
        send_message_tool = Tool(
            name="send_message_to_friend",
            description="Sends a message to one of the connected friend's agents to ask about their availability.",
            function=self.send_message_to_friend,
        )

        agent = Agent(
            instructions=system_prompt,
            tools=[
                list_courts_availability,
                book_court,
                send_message_tool,
            ],
        )
        print("  - System prompt configured.")
        print("  - Tools registered: list_courts_availability, book_court, send_message_to_friend")
        return agent

# --- Main execution block ---
async def main():
    # These would be the actual URLs of your running remote agent servers
    mock_remote_urls = [
        "http://localhost:8001", # Friend 1 (LangGraph)
        "http://localhost:8002", # Friend 2 (CrewAI)
        "http://localhost:8003", # Friend 3 (ADK)
    ]
    
    host = PickleballHostAgent(remote_agent_urls=mock_remote_urls)
    
    # Start a conversation
    print("-" * 50)
    print("Starting conversation with Host Agent...")
    print("Example Query: 'Find a time for me, Alice, and Bob to play pickleball this Friday evening.'")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = await host.adk_agent.chat(user_input)
            print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    asyncio.run(main())