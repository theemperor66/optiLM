import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API URL (configurable via environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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
        response = requests.post(
            f"{API_URL}/chat",
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

        return result
    except requests.RequestException as e:
        print(f"Error calling API: {str(e)}")
        return None

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
        response = requests.post(
            f"{API_URL}/chat",
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

        return result
    except requests.RequestException as e:
        print(f"Error calling API: {str(e)}")
        return None

if __name__ == "__main__":
    # Test regular mode
    test_chat_endpoint()

    # Test test mode
    test_chat_endpoint_with_test_mode()
