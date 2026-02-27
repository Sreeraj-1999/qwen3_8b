"""
Test if fine-tuned DeepSeek model can do tool calling
======================================================
This checks if your LLM can recognize and call tools in a structured format.
"""

import requests
import json

# Define mock tools your agent would have
TOOLS = [
    {
        "name": "get_fuel_status",
        "description": "Get fuel consumption prediction for a vessel",
        "parameters": {"vessel_imo": "string"}
    },
    {
        "name": "analyze_alarm",
        "description": "Analyze alarm and provide reasons/actions",
        "parameters": {"alarm_name": "string"}
    },
    {
        "name": "query_manual",
        "description": "Search vessel manuals for information",
        "parameters": {"question": "string"}
    }
]

def test_tool_calling(query: str):
    """Test if LLM can call tools"""

    # Format tools description
    tools_text = "\n".join([
        f"{i+1}. {t['name']}: {t['description']}"
        for i, t in enumerate(TOOLS)
    ])

    # Prompt that asks model to use tools
    prompt = f"""You are a marine engineering assistant with these tools:

{tools_text}

IMPORTANT: When you need to use a tool, respond EXACTLY in this format:
TOOL_CALL: {{"tool": "tool_name", "params": {{"param": "value"}}}}

If no tool is needed, just answer normally.

User: {query}
Assistant:"""

    messages = [{"role": "user", "content": prompt}]

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    # Call your LLM
    try:
        response = requests.post(
            "http://localhost:5005/gpu/llm/generate",
            json={"messages": messages, "response_type": "tool_test"},
            timeout=60
        )
        result = response.json().get('response', '')

        print(f"\nLLM Response:\n{result}\n")

        # Check if tool calling format detected
        has_tool_call = "TOOL_CALL:" in result or ('"tool"' in result and '"params"' in result)

        if has_tool_call:
            print("[+] Model attempted tool calling format")
            return True
        else:
            print("[-] No tool calling detected (answered directly)")
            return False

    except Exception as e:
        print(f"[X] Error: {e}")
        return False

# Test queries - should trigger tool calls
test_queries = [
    "What is the fuel consumption for vessel IMO9876543?",  # Should call get_fuel_status
    "Analyze high lube oil temperature alarm",              # Should call analyze_alarm
    "How to wire the power supply?",                        # Should call query_manual
    "What causes engine overheating?",                      # General Q&A - may not need tool
]

def run_tests():
    print("""
========================================================================
            TOOL CALLING CAPABILITY TEST
========================================================================

OBJECTIVE:
----------
Check if your fine-tuned DeepSeek can:
1. Detect when a tool should be called
2. Output tool calls in structured format
3. Choose correct tool for the query

WHAT THIS MEANS:
----------------
CAN do tool calling -> Build function-calling agents (LangGraph/CrewAI)
CANNOT do tool calling -> Use keyword routing (simpler, still powerful)

Testing with 4 sample queries...
""")

    # Run tests
    results = []
    for query in test_queries:
        result = test_tool_calling(query)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Tool calls detected: {sum(results)}/{len(results)}")
    print()

    if sum(results) >= 2:
        print("[+] YOUR MODEL CAN DO TOOL CALLING!")
        print("  -> You can build function-calling agents")
        print("  -> Use LangGraph or CrewAI with tool definitions")
        print("  -> Model will decide when to call tools")
    elif sum(results) == 1:
        print("[!] PARTIAL TOOL CALLING ABILITY")
        print("  -> Model might need more specific prompting")
        print("  -> Or use hybrid approach (keyword + LLM confirmation)")
    else:
        print("[-] MODEL PREFERS DIRECT ANSWERS")
        print("  -> Fine-tuned for Q&A, not tool calling")
        print("  -> Use keyword-based routing instead")
        print("  -> Example: if 'fuel' in query -> call fuel_analysis()")
        print("  -> Still works great for agents, just different architecture")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import sys
    print("Make sure gpu_service.py is running on port 5005")
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("Running in auto mode...\n")
    else:
        print("Press Enter to start testing...")
        input()
    run_tests()
