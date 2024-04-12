# This code is designed to summarize a text file paragraph by paragraph 
# using a pre-trained BART (Bidirectional and Auto-Regressive Transformer) model 
# from Hugging Face

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os

def summarize_paragraphs(input_file, output_file):
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')

    # Open input file and read paragraphs
    with open(input_file, "r") as file:
        paragraphs = file.read().split("\n\n")

    # Create an empty list to store summarized paragraphs
    summarized_paragraphs = []

    # Process each paragraph
    for i, paragraph in enumerate(paragraphs, 1):
        # Tokenize and encode the paragraph
        inputs = tokenizer(paragraph, return_tensors='pt')

        # Generate summary using BART model
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Check if paragraph contains the tag string '##'
        if '##' in paragraph and '##' not in summary:
            # Find the position of '##' in the paragraph
            tag_pos = paragraph.find('##')
            # Extract the tag from the paragraph
            tag = paragraph[tag_pos:paragraph.find(' ', tag_pos) if ' ' in paragraph[tag_pos:] else len(paragraph)]
            # Add the tag to the summary
            summary = tag + ' ' + summary

        # Append summarized paragraph to the list
        summarized_paragraphs.append(summary)

        # Display progress
        print(f"Paragraph {i}/{len(paragraphs)} processed")

    # Write all summarized paragraphs to a single output file
    with open(output_file, "w") as outfile:
        outfile.write("\n\n".join(summarized_paragraphs))

def main():
    # Ask user for input and output file paths
    input_file = input("Enter the path to the input text file: ")
    output_file = input("Enter the path to the output text file: ")
    
    # Check if the input file exists
    if not os.path.exists(input_file):
        print("Input file not found.")
        return
    
    # Summarize paragraphs
    summarize_paragraphs(input_file, output_file)

if __name__ == "__main__":
    main()
