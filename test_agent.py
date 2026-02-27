"""
Quick test script for Marine Agent
"""

from marine_agent import MarineAgent

def main():
    print("""
========================================================================
                    MARINE AGENT - QUICK TEST
========================================================================
    """)

    agent = MarineAgent(
        main_api_url="http://localhost:5004",
        gpu_api_url="http://localhost:5005"
    )

    # Test queries
    tests = [
        ("What is the fuel consumption?", "9876543"),
        ("Analyze high lube oil temperature alarm", "9876543"),
        ("How to wire the power supply?", "9876543"),
        ("What causes engine overheating?", None),
    ]

    print("Running 4 test queries...\n")

    for query, imo in tests:
        print(f"\nQuery: {query}")
        if imo:
            print(f"Vessel: IMO{imo}")

        result = agent.process_query(query, imo)

        print(f"→ Intent: {result['intent']}")
        print(f"→ Status: {result['status']}")

        if result.get('answer'):
            print(f"→ Answer: {result['answer'][:150]}...")

        print("-" * 70)

    print("\n✓ Test complete!")
    print("\nTo use interactively, run: python marine_agent.py")


if __name__ == "__main__":
    main()
