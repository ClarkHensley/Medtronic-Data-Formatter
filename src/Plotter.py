#!/usr/bin/env python3

"""
This file will plot the data after it has been extracted
"""

def main():

    # Psuedocode to follow
    # Generate the Settings and Dataset/Groups dictionaries
    # If the extracted files don't exist, inform the user, and begin the extraction
    # Otherwise, prompt them to extract (default No)
    # After extractions, we'll open each CSV one-at-a-time (By Group, then by Test)
    # first, process each strike segment (use threshold and iterations specified in the groups dictionary)
    # We'll have to handle the main dataset from this file too (HOW?), to manage things like initialAppend (Will we?)
    # Here, we'll plot each strike, We can do one strike per plot, or all per a test (or both). We'll try to give the user a number of options (Including saving to a new name so as to not overwrite the previous generated images with the same data)
    # For each group, we plot the violin plots
    # Then, for the dataset, plot the mean violin plots
    # Store the final means, stdevs, etc in the final CSV
    # Call it done

    # So, we'll generate some things (settings, dataset/groups dictionaries, and maybe the total_dataset, and pass that into extractFromAll(), which is built to recreate those if they don't exist.)

    pass

if __name__ == "__main__":
    main()
