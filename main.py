from init import init
from flock_sdk import FlockSDK
from arguments import load_arguments

if __name__ == "__main__":
    args = load_arguments()
    task_model = init(args)
    sdk = FlockSDK(task_model)
    sdk.run()
