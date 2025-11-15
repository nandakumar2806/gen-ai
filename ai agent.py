import os
import logging
from typing import Any, Dict, List, Callable, Optional
from abc import ABC, abstractmethod


def get_config(param: str, default: Optional[str] = None) -> str:
    """
    Retrieve configuration from environment variables.
    
    Args:
        param (str): Configuration parameter name.
        default (Optional[str]): Default value if not found.
    
    Returns:
        str: Configuration value.
    
    Raises:
        EnvironmentError: If parameter not found and no default provided.
    """
    val = os.environ.get(param, default)
    if val is None:
        raise EnvironmentError(f"Missing required configuration: {param}")
    return val


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Sets up logging format and level for observability.
    
    Args:
        log_level (int): Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


# Initialize logging
setup_logging()
logger = logging.getLogger("AI_Agent_Assistant")


def summarize_text(text: str) -> str:
    """
    Summarize text by extracting first sentence.
    
    Args:
        text (str): Text to summarize.
    
    Returns:
        str: Summarized text.
    """
    if not text:
        logger.warning("No text provided to summarize_text tool.")
        return ""
    summary = text.split(".")[0] + "." if "." in text else text
    logger.info("Summarized text.")
    return summary


class BaseAgent(ABC):
    """
    Abstract base agent with core properties and memory/context.
    Demonstrates agent architecture and memory management.
    """
    
    def __init__(self, name: str, tools: List[Callable[..., Any]]):
        """
        Initialize base agent.
        
        Args:
            name (str): Agent identifier.
            tools (List[Callable]): List of capabilities/tools.
        """
        self.name = name
        self.tools = tools
        self.memory: List[str] = []
        logger.info(f"Initialized agent: {self.name}")

    def remember(self, info: str) -> None:
        """
        Store context in agent memory.
        
        Args:
            info (str): Information to store.
        """
        self.memory.append(info)
        logger.debug(f"{self.name} remembered: {info}")

    def recall(self) -> List[str]:
        """
        Retrieve memory contents.
        
        Returns:
            List[str]: All stored memories.
        """
        logger.debug(f"{self.name} recalling memory.")
        return self.memory

    @abstractmethod
    def act(self, *args, **kwargs) -> Any:
        """
        Define agent action logic in subclasses.
        
        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError("Override this method in subclasses.")


class MultiAgentSystem:
    """
    Controls and coordinates interactions between multiple agents.
    Demonstrates multi-agent orchestration and sequential execution.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize multi-agent system.
        
        Args:
            agents (List[BaseAgent]): List of agents to orchestrate.
        """
        self.agents = agents
        self.trace: List[Dict[str, Any]] = []
        logger.info(f"Initialized MultiAgentSystem with {len(agents)} agents")

    def run_sequential(self, task_input: Any) -> Any:
        """
        Passes input through agents in sequence.
        
        Args:
            task_input (Any): Initial input for agent chain.
        
        Returns:
            Any: Final output after all agents act.
        """
        logger.info("Running multi-agent sequential pipeline.")
        output = task_input
        for agent in self.agents:
            logger.info(f"Agent {agent.name} acting...")
            try:
                output = agent.act(output)
                self.trace.append({"agent": agent.name, "output": output})
                logger.debug(f"Agent {agent.name} output: {output}")
            except Exception as e:
                logger.error(f"Error in agent {agent.name}: {str(e)}")
                raise
        return output

    def run_parallel(self, task_input: Any) -> Dict[str, Any]:
        """
        Run all agents in parallel with same input.
        
        Args:
            task_input (Any): Input for all agents.
        
        Returns:
            Dict[str, Any]: Results from each agent.
        """
        logger.info("Running agents in parallel.")
        results = {}
        for agent in self.agents:
            logger.info(f"Agent {agent.name} acting (parallel)...")
            try:
                output = agent.act(task_input)
                results[agent.name] = output
                self.trace.append({"agent": agent.name, "output": output})
            except Exception as e:
                logger.error(f"Error in parallel agent {agent.name}: {str(e)}")
                results[agent.name] = None
        return results

    def log_trace(self) -> None:
        """Log execution trace."""
        logger.info("Orchestration Trace:")
        for step in self.trace:
            logger.info(f"  Agent: {step['agent']}, Output: {step['output']}")

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get execution trace.
        
        Returns:
            List[Dict[str, Any]]: Execution history.
        """
        return self.trace.copy()


class ResearchAgent(BaseAgent):
    """
    Example agent to perform research via summarization.
    Implements sequential chain concept with memory.
    """
    
    def act(self, query: str) -> str:
        """
        Handles research task (placeholder logic).
        
        Args:
            query (str): Research prompt/question.
        
        Returns:
            str: Research output (summary).
        """
        logger.info(f"ResearchAgent processing query: {query[:50]}...")
        result = summarize_text(query)
        self.remember(result)
        return result


class AnalysisAgent(BaseAgent):
    """
    Agent for analyzing and processing information.
    Demonstrates sequential chaining capability.
    """
    
    def act(self, data: str) -> str:
        """
        Analyze provided data.
        
        Args:
            data (str): Data to analyze.
        
        Returns:
            str: Analysis result.
        """
        logger.info(f"AnalysisAgent processing data: {data[:50]}...")
        # Placeholder analysis logic
        analysis = f"Analysis of: {data}"
        self.remember(analysis)
        return analysis


class EvaluationAgent(BaseAgent):
    """
    Agent to evaluate results (demonstrates evaluation concept).
    Final agent in the pipeline for quality assurance.
    """
    
    def act(self, output: str) -> str:
        """
        Scores or critiques agent-generated output.
        
        Args:
            output (str): Output to evaluate.
        
        Returns:
            str: Evaluation result (placeholder logic).
        """
        logger.info(f"EvaluationAgent evaluating: {output[:50]}...")
        if len(output) > 50:
            verdict = "Output is detailed."
        else:
            verdict = "Output may lack detail."
        self.remember(verdict)
        logger.info(f"EvaluationAgent verdict: {verdict}")
        return verdict


def main():
    """Main execution function."""
    print(
        "Hello! I'm your AI Agents Capstone Project developer assistant.\n"
        "I'm here to help you build a production-ready AI agent that demonstrates "
        "the concepts you learned in the 5-Day AI Agents Intensive Course.\n\n"
        "Let's start with understanding your project:\n"
        "1. What problem are you trying to solve?\n"
        "2. Which track interests you? (Concierge, Research, Automation, etc.)\n"
        "3. Do you have any initial ideas on how AI agents could help?\n"
        "4. How much time do you have to develop this?\n\n"
        "The more details you share, the better I can help you design and build "
        "a winning capstone project! 🚀\n"
        "Remember: Your goal is to help create a project that clearly demonstrates "
        "course mastery while delivering real value. Happy building!"
    )

    print("\n" + "="*80)
    print("RUNNING DEMO: Sequential Multi-Agent Pipeline")
    print("="*80 + "\n")

    user_query = "Explain the basics of reinforcement learning. It is a paradigm where agents learn by trial and error."
    research_agent = ResearchAgent("ResearchAgent", [summarize_text])
    evaluation_agent = EvaluationAgent("EvaluationAgent", [])
    system = MultiAgentSystem([research_agent, evaluation_agent])
    
    final_output = system.run_sequential(user_query)
    system.log_trace()
    print("Final Evaluation:", final_output)


def test_research_and_evaluation():
    """
    Unit test demonstrating core agent pipeline.
    """
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80 + "\n")
    
    qa = "What is supervised learning? It involves training a model on labeled data."
    agent1 = ResearchAgent("ResearchTest", [summarize_text])
    agent2 = EvaluationAgent("EvalTest", [])
    mas = MultiAgentSystem([agent1, agent2])
    
    result = mas.run_sequential(qa)
    
    # Assertions
    assert isinstance(result, str), "Result should be a string"
    assert agent1.memory, "ResearchAgent should store memory."
    assert agent2.memory, "EvaluationAgent should store memory."
    
    print("✓ Test passed: Research and evaluation flow works.")
    print(f"  - Result type: {type(result).__name__}")
    print(f"  - Agent1 memory items: {len(agent1.memory)}")
    print(f"  - Agent2 memory items: {len(agent2.memory)}")


if __name__ == "__main__":
    main()
    test_research_and_evaluation()
