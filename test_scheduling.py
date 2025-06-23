from fastapi.testclient import TestClient
from api.main import app
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set dummy API key for tests
os.environ.setdefault("GOOGLE_API_KEY", "test")

# Use FastAPI TestClient for in-process testing
client = TestClient(app)

# API URL (configurable via environment variable)

def test_chat_endpoint():
    """Test the chat endpoint with a scheduling problem."""
    print("Testing chat endpoint with a scheduling problem...")

    # Example scheduling problem description
    message = """
    I have 2 machines and 3 jobs. 
    Job 1 requires rig 1 and has a processing time of 2, Job 2 requires rig 2 and has a processing time of 3, 
    and Job 3 requires rig 1 and has a processing time of 1.
    The rig change time from rig 1 to rig 2 is 2 units, and from rig 2 to rig 1 is 1 unit.
    Use the GLOBAL solver with a maximum time of 30 seconds and enable heuristics.
    """

    # Call the API
    try:
        response = client.post(
            "/chat",
            json={"message": message}
        )
        response.raise_for_status()
        result = response.json()

        # Print the response
        print("\nAPI Response:")
        print(f"Response message: {result['response']}")
        print(f"Requires support: {result['requires_support']}")

        # Check if a scheduling problem was formulated
        if result.get('scheduling_problem'):
            print("\nScheduling Problem:")
            print(json.dumps(result['scheduling_problem'], indent=2))
        else:
            print("\nNo scheduling problem was formulated.")

        # Check if the API was called and returned a response
        if result.get('api_response'):
            print("\nAPI Response:")
            print(json.dumps(result['api_response'], indent=2))
        else:
            print("\nNo API response was returned.")

        assert result is not None
        assert 'response' in result
    except Exception as e:
        raise AssertionError(f"Error calling API: {e}")

def test_chat_endpoint_with_test_mode():
    """Test the chat endpoint with a scheduling problem in test mode."""
    print("\nTesting chat endpoint with a scheduling problem in test mode...")

    # Example scheduling problem description
    message = """
    I have 2 machines and 3 jobs. 
    Job 1 requires rig 1 and has a processing time of 2, Job 2 requires rig 2 and has a processing time of 3, 
    and Job 3 requires rig 1 and has a processing time of 1.
    The rig change time from rig 1 to rig 2 is 2 units, and from rig 2 to rig 1 is 1 unit.
    Use the GLOBAL solver with a maximum time of 30 seconds and enable heuristics.
    """

    # Call the API with test_mode=True
    try:
        response = client.post(
            "/chat",
            json={"message": message, "test_mode": True}
        )
        response.raise_for_status()
        result = response.json()

        # Print the response
        print("\nAPI Response (Test Mode):")
        print(f"Response message: {result['response']}")
        print(f"Requires support: {result['requires_support']}")

        # Check if a scheduling problem was formulated
        if result.get('scheduling_problem'):
            print("\nScheduling Problem:")
            print(json.dumps(result['scheduling_problem'], indent=2))
        else:
            print("\nNo scheduling problem was formulated.")

        # Check if the API was called and returned a response
        if result.get('api_response'):
            print("\nAPI Response (Random Solution):")
            print(json.dumps(result['api_response'], indent=2))

            # Verify that this is a random solution by checking the status and solution structure
            if result['api_response']['status'] == 'success' and 'solution' in result['api_response']:
                print("\nTest mode is working correctly! Random solution was generated.")

                # Verify that every machine has start_rig_id
                if result.get('scheduling_problem') and result['scheduling_problem'].get('machines'):
                    all_machines_have_start_rig = all('start_rig_id' in m for m in result['scheduling_problem']['machines'])
                    if all_machines_have_start_rig:
                        print("All machines have start_rig_id field ✓")
                    else:
                        print("ERROR: Not all machines have start_rig_id field!")
            else:
                print("\nTest mode may not be working correctly. Check the response structure.")
        else:
            print("\nNo API response was returned.")

        assert result is not None
        assert 'response' in result
    except Exception as e:
        raise AssertionError(f"Error calling API: {e}")

def test_conversational_approach():
    """Test the conversational approach with multiple messages."""
    print("\nTesting conversational approach with multiple messages...")

    # Step 1: Start with machines
    message1 = "I have 2 machines"

    try:
        # First message - should ask about jobs
        response1 = client.post(
            "/chat",
            json={"message": message1, "test_mode": True}
        )
        response1.raise_for_status()
        result1 = response1.json()

        print("\nStep 1 - Machines:")
        print(f"Response: {result1['response']}")
        print(f"Is problem complete: {result1.get('is_problem_complete', False)}")
        print(f"Full response: {result1}")

        # Get the context from the first response
        context1 = result1.get('scheduling_problem', {})

        # Step 2: Add jobs
        message2 = "Job 1 rig 1 time 3; Job 2 rig 2 time 4"

        response2 = client.post(
            "/chat",
            json={"message": message2, "context": context1, "test_mode": True}
        )
        response2.raise_for_status()
        result2 = response2.json()

        print("\nStep 2 - Jobs:")
        print(f"Response: {result2['response']}")
        print(f"Is problem complete: {result2.get('is_problem_complete', False)}")

        # Get the context from the second response
        context2 = result2.get('scheduling_problem', {})

        # Step 3: Add rig matrix and max time
        message3 = "rig matrix [[0,1],[1,0]]; max time 30"

        response3 = client.post(
            "/chat",
            json={"message": message3, "context": context2, "test_mode": True}
        )
        response3.raise_for_status()
        result3 = response3.json()

        print("\nStep 3 - Rig Matrix:")
        print(f"Response: {result3['response']}")
        print(f"Is problem complete: {result3.get('is_problem_complete', False)}")

        # Get the context from the third response
        context3 = result3.get('scheduling_problem', {})

        # Step 4: Solve the problem
        message4 = "solve"

        response4 = client.post(
            "/chat",
            json={"message": message4, "context": context3, "test_mode": True}
        )
        response4.raise_for_status()
        result4 = response4.json()

        print("\nStep 4 - Solve:")
        print(f"Response: {result4['response']}")
        print(f"Is problem complete: {result4.get('is_problem_complete', False)}")

        # Check if the API was called and returned a response
        if result4.get('api_response') and result4['api_response'].get('status') == 'success':
            print("\nTest passed! Conversational approach is working correctly.")
            assert True
        else:
            print("\nTest failed! API response not received or status not success.")
            assert False

    except Exception as e:
        raise AssertionError(f"Error calling API: {e}")

if __name__ == "__main__":
    # Test regular mode
    # test_chat_endpoint()

    # Test test mode
    # test_chat_endpoint_with_test_mode()

    # Test conversational approach
    test_conversational_approach()
