#!/usr/bin/env python3

while (True):
    exit_keys = ["exit", "quit", "goodbye", "bye"]
    question = input("Q: ")

    print("A: ", end="")
    if question.strip().lower() in exit_keys:
        print("Goodbye")
        break
    else:
        print()
