#!/usr/bin/env python3
"""
script that takes in input from the user with the prompt Q: and prints A: as a response.
"""
while (True):
    exit_keys = ["exit", "quit", "goodbye", "bye"]
    question = input("Q: ")

    print("A: ", end="")
    if question.strip().lower() in exit_keys:
        print("Goodbye")
        break
    else:
        print()
