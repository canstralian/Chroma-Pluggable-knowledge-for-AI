import tkinter as tk
from tkinter import messagebox

class MyApplication:
    def __init__(self, master):
        self.master = master
        master.title("My Application")
        master.geometry("400x300")

        self.label = tk.Label(master, text="Welcome to My Application")
        self.entry = tk.Entry(master)
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_action)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear_action)
        self.result_label = tk.Label(master, text="")

        self.label.pack()
        self.entry.pack()
        self.submit_button.pack(side=tk.LEFT, padx=5)
        self.clear_button.pack(side=tk.RIGHT, padx=5)
        self.result_label.pack()

    def submit_action(self):
        input_text = self.entry.get()
        if input_text:
            processed_result = self.process_input(input_text)
            self.result_label.config(text="Result: " + processed_result)
        else:
            messagebox.showerror("Error", "Please enter some text.")

    def clear_action(self):
        self.entry.delete(0, tk.END)
        self.result_label.config(text="")

    def process_input(self, text):
        # Placeholder function for processing input
        return text.upper()  # Example processing: convert text to uppercase

def main():
    root = tk.Tk()
    app = MyApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()