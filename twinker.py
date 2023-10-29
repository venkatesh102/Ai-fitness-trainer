import tkinter as tk
from functools import partial

# ... (Your Module class and speech functions here)

class FitnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitness Trainer App")

        self.label = tk.Label(root, text="Welcome to the Fitness Trainer App")
        self.label.pack()

        exercises = ["Right Bicep Curl", "Left Bicep Curl", "Pushups", "Squats", "Skipping"]

        for i, exercise in enumerate(exercises, start=1):
            button = tk.Button(root, text=f"{i}. {exercise}", command=partial(self.start_exercise, i))
            button.pack()

    def start_exercise(self, exercise_id):
        if exercise_id == 1:
            right_bicep_curl()
        elif exercise_id == 2:
            left_bicep_curl()
        elif exercise_id == 3:
            pushups()
        elif exercise_id == 4:
            squats()
        elif exercise_id == 5:
            skipping()

if __name__ == "__main":
    root = tk.Tk()
    app = FitnessApp(root)
    root.mainloop()
