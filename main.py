import subprocess

def run_task(task_name):
    """ Helper function to run a task script via subprocess """
    print(f"Running {task_name}...")
    result = subprocess.run(["python3", task_name], capture_output=True, text=True)
    print(result.stdout)
    print(f"{task_name} completed.\n")

if __name__ == "__main__":
    # Run each task by calling the task scripts
    run_task("task1.py")  # This will run the task1 script
    run_task("task2.py")  # This will run the task2 script
    run_task("task3.py")  # This will run the task3 script
