"""
Marine Engineering Agent System
================================
Intent-based routing using LLM classification instead of simple keywords.

Features:
- Smart intent classification using your fine-tuned LLM
- Integrates with existing APIs (fuel, alarm, manual, maintenance)
- Handles vessel-specific operations
- Fallback to general Q&A
"""

import requests
import json
import logging
from typing import Dict, Optional, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarineAgent:
    """
    Main agent that routes queries to appropriate tools based on LLM intent classification.
    """

    def __init__(self,
                 main_api_url: str = "http://localhost:5004",
                 gpu_api_url: str = "http://localhost:5005"):
        self.main_api = main_api_url
        self.gpu_api = gpu_api_url

        # Available intents and their descriptions (order matters for matching)
        self.intents = {
            "ALARM_ANALYSIS": "Questions about specific alarms, warnings, troubleshooting alarm conditions (e.g., 'analyze high temperature alarm')",
            "MANUAL_QUERY": "Questions needing manual/documentation lookup - how-to guides, procedures, specifications, wiring instructions",
            "FUEL_ANALYSIS": "Questions about fuel consumption, efficiency, predictions, or fuel-related performance metrics",
            "MAINTENANCE_PREDICTION": "Questions about predictive maintenance, component failure predictions, smart maintenance analysis",
            "GENERAL_QA": "General marine engineering knowledge questions that can be answered from basic knowledge (e.g., 'what causes overheating')"
        }

    def classify_intent(self, query: str) -> str:
        """
        Use LLM to classify user intent instead of keyword matching.
        This is more accurate and handles natural language better.
        """

        # Build intent descriptions
        intent_list = "\n".join([
            f"- {intent}: {desc}"
            for intent, desc in self.intents.items()
        ])

        prompt = f"""Classify this marine engineering query into exactly ONE category.

Categories:
{intent_list}

Query: "{query}"

Rules:
1. Output ONLY the category name (e.g., ALARM_ANALYSIS)
2. NO explanations, NO extra text
3. If query mentions specific alarm → ALARM_ANALYSIS
4. If asking "how to" or needs manual → MANUAL_QUERY
5. If about fuel/consumption → FUEL_ANALYSIS

Your answer (category name only):"""

        try:
            response = requests.post(
                f"{self.gpu_api}/gpu/llm/generate",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "response_type": "intent_classification"
                },
                timeout=30
            )

            raw_response = response.json().get('response', 'GENERAL_QA').strip()

            # Extract only the intent - model sometimes adds extra text
            # Look for the first line or first valid intent keyword
            first_line = raw_response.split('\n')[0].strip().upper()

            # Check if any valid intent is in the first line
            for valid_intent in self.intents.keys():
                if valid_intent in first_line:
                    logger.info(f"Classified intent: {valid_intent} (from: {first_line[:50]})")
                    return valid_intent

            # Fallback: check entire response
            for valid_intent in self.intents.keys():
                if valid_intent in raw_response.upper():
                    logger.info(f"Classified intent: {valid_intent} (fallback match)")
                    return valid_intent

            logger.warning(f"Unknown intent '{first_line[:50]}', defaulting to GENERAL_QA")
            return "GENERAL_QA"

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "GENERAL_QA"

    def extract_vessel_imo(self, query: str) -> Optional[str]:
        """Extract IMO number from query if present."""
        import re
        match = re.search(r'IMO\s*(\d+)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def extract_alarm_name(self, query: str) -> Optional[str]:
        """Extract alarm name from query using LLM."""
        prompt = f"""Extract the alarm name from this query. If no specific alarm is mentioned, respond with 'NONE'.

Query: {query}

Alarm name (or NONE):"""

        try:
            response = requests.post(
                f"{self.gpu_api}/gpu/llm/generate",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "response_type": "entity_extraction"
                },
                timeout=20
            )
            alarm = response.json().get('response', '').strip()
            return None if alarm.upper() == 'NONE' else alarm
        except:
            return None

    # ======================== TOOL FUNCTIONS ========================

    def fuel_analysis_tool(self, query: str, vessel_imo: Optional[str] = None) -> Dict:
        """
        Call fuel analysis API.
        Note: This needs actual sensor data payload - here we show the structure.
        """
        logger.info(f"Calling fuel analysis tool for IMO: {vessel_imo}")

        # In production, you'd get real sensor data
        # For now, return instruction on how to use it
        return {
            "status": "info",
            "message": f"To analyze fuel consumption for vessel {vessel_imo or 'unknown'}, I need real-time sensor data including SOG, STW, RPM, torque, power, wind direction, and ship heading.",
            "endpoint": f"{self.main_api}/fuel/analysis",
            "query": query
        }

    def alarm_analysis_tool(self, query: str, vessel_imo: Optional[str] = None) -> Dict:
        """Call alarm analysis API."""
        logger.info(f"Calling alarm analysis tool")

        # Extract alarm name
        alarm_name = self.extract_alarm_name(query)

        if not alarm_name:
            return {
                "status": "error",
                "message": "Could not identify specific alarm name. Please specify the alarm you want to analyze."
            }

        if not vessel_imo:
            return {
                "status": "error",
                "message": "Vessel IMO is required for alarm analysis. Please specify the vessel."
            }

        try:
            response = requests.post(
                f"{self.main_api}/vessels/{vessel_imo}/alarms/analyze",
                json={"alarm_name": [alarm_name]},
                timeout=60
            )

            result = response.json()
            return {
                "status": "success",
                "alarm_name": alarm_name,
                "vessel_imo": vessel_imo,
                "analysis": result.get('data', []),
                "query": query
            }
        except Exception as e:
            logger.error(f"Alarm analysis failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to analyze alarm: {str(e)}"
            }

    def manual_query_tool(self, query: str, vessel_imo: Optional[str] = None) -> Dict:
        """Call manual query API (RAG)."""
        logger.info(f"Calling manual query tool for IMO: {vessel_imo}")

        if not vessel_imo:
            # General chat without vessel context
            return self.general_qa_tool(query)

        try:
            response = requests.post(
                f"{self.main_api}/vessels/{vessel_imo}/manuals/query",
                json={"question": query},
                timeout=60
            )

            result = response.json()
            return {
                "status": "success",
                "vessel_imo": vessel_imo,
                "answer": result.get('answer', ''),
                "source": result.get('source', ''),
                "metadata": result.get('metadata', []),
                "query": query
            }
        except Exception as e:
            logger.error(f"Manual query failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to query manuals: {str(e)}"
            }

    def maintenance_prediction_tool(self, query: str, vessel_imo: Optional[str] = None) -> Dict:
        """Call predictive maintenance API."""
        logger.info(f"Calling maintenance prediction tool")

        try:
            response = requests.post(
                f"{self.main_api}/predict",
                timeout=120
            )

            result = response.json()
            return {
                "status": "success",
                "prediction": result,
                "query": query
            }
        except Exception as e:
            logger.error(f"Maintenance prediction failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to predict maintenance: {str(e)}"
            }

    def general_qa_tool(self, query: str) -> Dict:
        """Call general Q&A (no vessel context, no RAG)."""
        logger.info(f"Calling general Q&A tool")

        try:
            response = requests.post(
                f"{self.main_api}/chat/response/",
                json={"chat": query},
                timeout=60
            )

            result = response.json()
            return {
                "status": "success",
                "answer": result.get('data', {}).get('answer', ''),
                "source": "llm_knowledge",
                "query": query
            }
        except Exception as e:
            logger.error(f"General Q&A failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to get answer: {str(e)}"
            }

    # ======================== MAIN AGENT LOGIC ========================

    def process_query(self, query: str, vessel_imo: Optional[str] = None) -> Dict:
        """
        Main agent entry point.
        1. Classify intent using LLM
        2. Extract entities (IMO, alarm name, etc.)
        3. Route to appropriate tool
        4. Return result
        """

        logger.info(f"Processing query: {query}")

        # Extract IMO if not provided
        if not vessel_imo:
            vessel_imo = self.extract_vessel_imo(query)

        # Classify intent
        intent = self.classify_intent(query)

        # Route to appropriate tool
        if intent == "FUEL_ANALYSIS":
            result = self.fuel_analysis_tool(query, vessel_imo)

        elif intent == "ALARM_ANALYSIS":
            result = self.alarm_analysis_tool(query, vessel_imo)

        elif intent == "MANUAL_QUERY":
            result = self.manual_query_tool(query, vessel_imo)

        elif intent == "MAINTENANCE_PREDICTION":
            result = self.maintenance_prediction_tool(query, vessel_imo)

        else:  # GENERAL_QA
            result = self.general_qa_tool(query)

        # Add metadata
        result['intent'] = intent
        result['vessel_imo'] = vessel_imo

        return result

    def chat(self, query: str, vessel_imo: Optional[str] = None, verbose: bool = True) -> str:
        """
        User-friendly chat interface.
        Returns just the answer string.
        """
        result = self.process_query(query, vessel_imo)

        if verbose:
            print(f"\n[Intent: {result.get('intent', 'unknown')}]")
            if vessel_imo:
                print(f"[Vessel: IMO{vessel_imo}]")

        if result.get('status') == 'success':
            return result.get('answer', result.get('message', str(result)))
        else:
            return f"Error: {result.get('message', 'Unknown error')}"


# ======================== TESTING ========================

def test_agent():
    """Test the agent with various queries."""

    print("""
========================================================================
                    MARINE AGENT SYSTEM TEST
========================================================================

Testing intent-based routing with your fine-tuned LLM...
""")

    agent = MarineAgent()

    # Test cases
    test_cases = [
        {
            "query": "What is the fuel consumption for vessel IMO9876543?",
            "vessel_imo": "9876543",
            "expected_intent": "FUEL_ANALYSIS"
        },
        {
            "query": "Analyze the high lube oil temperature alarm for IMO9876543",
            "vessel_imo": "9876543",
            "expected_intent": "ALARM_ANALYSIS"
        },
        {
            "query": "How do I wire the power supply?",
            "vessel_imo": "9876543",
            "expected_intent": "MANUAL_QUERY"
        },
        {
            "query": "What causes engine overheating?",
            "vessel_imo": None,
            "expected_intent": "GENERAL_QA"
        },
        {
            "query": "Run predictive maintenance analysis",
            "vessel_imo": None,
            "expected_intent": "MAINTENANCE_PREDICTION"
        }
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {test['query']}")
        print(f"{'='*70}")

        result = agent.process_query(
            query=test['query'],
            vessel_imo=test['vessel_imo']
        )

        detected_intent = result.get('intent', 'UNKNOWN')
        expected_intent = test['expected_intent']
        correct = detected_intent == expected_intent

        print(f"Expected Intent: {expected_intent}")
        print(f"Detected Intent: {detected_intent}")
        print(f"Status: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
        print(f"\nResult Summary:")
        print(f"  Status: {result.get('status', 'unknown')}")

        if result.get('answer'):
            print(f"  Answer: {result['answer'][:100]}...")
        elif result.get('message'):
            print(f"  Message: {result['message'][:100]}...")

        results.append(correct)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY:")
    print(f"{'='*70}")
    print(f"Correct intent classifications: {sum(results)}/{len(results)}")
    print(f"Accuracy: {sum(results)/len(results)*100:.1f}%")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_agent()
    else:
        # Interactive mode
        print("Marine Agent System - Interactive Mode")
        print("Type 'quit' to exit\n")

        agent = MarineAgent()

        while True:
            query = input("\nYour question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            vessel_imo = input("Vessel IMO (press Enter to skip): ").strip() or None

            print("\nAgent Response:")
            print("-" * 70)
            answer = agent.chat(query, vessel_imo, verbose=True)
            print(answer)
            print("-" * 70)
