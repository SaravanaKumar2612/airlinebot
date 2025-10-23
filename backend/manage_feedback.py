#!/usr/bin/env python3
"""
Human Agent Feedback Management Tool
This script helps human agents review and label incorrect predictions for model retraining.
"""

import pandas as pd
import os

def view_feedback_data():
    """View all feedback data"""
    if not os.path.exists("feedback_log.csv"):
        print("No feedback data found.")
        return
    
    df = pd.read_csv("feedback_log.csv")
    print(f"\nTotal feedback entries: {len(df)}")
    print(f"Incorrect predictions: {len(df[df['correct'] == False])}")
    print(f"Correct predictions: {len(df[df['correct'] == True])}")
    
    # Show incorrect predictions that need labeling
    incorrect = df[df['correct'] == False]
    unlabeled = incorrect[incorrect['true_label'].isna() | (incorrect['true_label'] == '')]
    
    print(f"\nUnlabeled incorrect predictions: {len(unlabeled)}")
    
    if len(unlabeled) > 0:
        print("\nUnlabeled entries:")
        for idx, row in unlabeled.iterrows():
            print(f"\n{idx + 1}. Text: '{row['text']}'")
            print(f"   Predicted: {row['predicted_intent']}")
            print(f"   True Label: {row['true_label']}")

def add_true_labels():
    """Interactive tool to add true labels"""
    if not os.path.exists("feedback_log.csv"):
        print("No feedback data found.")
        return
    
    df = pd.read_csv("feedback_log.csv")
    incorrect = df[df['correct'] == False]
    unlabeled = incorrect[incorrect['true_label'].isna() | (incorrect['true_label'] == '')]
    
    if len(unlabeled) == 0:
        print("All incorrect predictions are already labeled!")
        return
    
    print(f"\nFound {len(unlabeled)} unlabeled incorrect predictions.")
    print("Available intent categories:")
    
    # Get all unique intents from the original model
    intents = [
        "Prohibited Items Faq", "Greetings/Special Assistance", "Insurance", "Cancel Trip",
        "Flights Info", "Airport Lounge Access", "Discounts", "Meal Preferences",
        "Check In Luggage Faq", "Travel Documents", "Carry On Luggage Faq",
        "Flight Status", "Missing Bag", "Complaints", "Fare Check", "Pet Travel",
        "Change Flight", "Frequent Flyer", "Travel Alerts", "Seat Availability",
        "Cancellation Policy", "Airport Transfers", "Baggage Delay", "Seat Upgrade Request",
        "Sports Music Gear", "Damaged Bag", "Medical Policy", "Refund / Compensation"
    ]
    
    for i, intent in enumerate(intents, 1):
        print(f"{i:2d}. {intent}")
    
    print("\nEnter 'skip' to skip an entry, 'quit' to exit")
    
    for idx, row in unlabeled.iterrows():
        print(f"\n--- Entry {idx + 1} ---")
        print(f"Text: '{row['text']}'")
        print(f"Predicted: {row['predicted_intent']}")
        
        while True:
            choice = input("Enter true label (number or name, or 'skip'): ").strip()
            
            if choice.lower() == 'quit':
                df.to_csv("feedback_log.csv", index=False)
                print("Changes saved. Exiting...")
                return
            elif choice.lower() == 'skip':
                print("Skipped.")
                break
            elif choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(intents):
                    df.at[idx, 'true_label'] = intents[choice_num - 1]
                    print(f"Labeled as: {intents[choice_num - 1]}")
                    break
                else:
                    print("Invalid number. Please try again.")
            else:
                # Check if it's a valid intent name
                if choice in intents:
                    df.at[idx, 'true_label'] = choice
                    print(f"Labeled as: {choice}")
                    break
                else:
                    print("Invalid intent name. Please try again.")
    
    # Save changes
    df.to_csv("feedback_log.csv", index=False)
    print("\nAll changes saved!")

def check_retraining_status():
    """Check if retraining threshold is reached"""
    if not os.path.exists("feedback_log.csv"):
        print("No feedback data found.")
        return
    
    df = pd.read_csv("feedback_log.csv")
    incorrect_count = len(df[df['correct'] == False])
    labeled_count = len(df[(df['correct'] == False) & (df['true_label'].notna()) & (df['true_label'] != '')])
    
    print(f"\nRetraining Status:")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Labeled for retraining: {labeled_count}")
    print(f"Threshold (50): {'✓ REACHED' if incorrect_count >= 50 else 'Not reached'}")
    
    if incorrect_count >= 50 and labeled_count >= 10:
        print("✓ Ready for retraining!")
    elif incorrect_count >= 50:
        print("⚠ Threshold reached but need more labeled data (minimum 10)")
    else:
        print(f"Need {50 - incorrect_count} more incorrect predictions")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("Human Agent Feedback Management Tool")
        print("="*50)
        print("1. View feedback data")
        print("2. Add true labels")
        print("3. Check retraining status")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            view_feedback_data()
        elif choice == '2':
            add_true_labels()
        elif choice == '3':
            check_retraining_status()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
