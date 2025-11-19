"""
Dataset Preprocessing Script - Version 3

Table of Contents:
===================

Classes:
--------
1. DatasetPreprocessor
1.1 MyHandler (xml.sax.ContentHandler)

Functions in class:
--------
1. _load_in_csv()
2. _search_xml_for_ids()
3. dataframe_found_entries()
4. extend_with_seq()
"""

# -------------------------
# 1. Imports
# -------------------------
import os
import time
import xml.sax

import pandas as pd

# -------------------------
# 2. GLOBALS
# -------------------------
XML_PATH = "/global/scratch2/sapelt/Protein_matched_complete/Protein_match_complete.xml"
CSV_PATH = "/global/scratch2/sapelt/Protein_matched_complete/uniprot_full.csv"
OUTPUT_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesCompleteProteins_tax.csv"
ID_COLUMN = "ID"
SEQUENCE_COLUMN = "Sequences"

# -------------------------
# 3. DatasetPreprocessor Class
# -------------------------


class DatasetPreprocessor:
    """
    Main class for preprocessing the dataset by extracting relevant entries from the protein_match_complete XML file given by pfam.
    It matches IDs from all uniprot protein sequences (SwissProt + TrEMBL) and extracts corresponding sequence and domain information form the XML.
    For this purpose, it uses a streaming XML parser searching for all important domain information.
    Those informations start with the 'lcn' tags inside 'match' tags for each protein entry.
    Since we only search for pfam domains, only matches with IDs starting with 'PF' are considered.

    Args:
        input_path_xml (str): Path to the large XML file containing protein and corresponding domain data.
        input_path_csv (str): Path to the CSV file containing target IDs and sequences.
    Returns:
        A CSV file saved at OUTPUT_PATH containing found entries with columns:
        "start", "end", "id", "Pfam_id", "taxid", "Sequence"
    """

    def __init__(self, input_path_xml, input_path_csv):
        """
        All functions are called here.
        Has early abort if output path already exists.
        """

        # Initialize paths & start print
        self.input_path_xml = input_path_xml
        self.input_path_csv = input_path_csv
        print("Starting dataset preprocessing...")

        # Check if output path already exists, otherwise abort early
        if os.path.exists(OUTPUT_PATH):
            print(
                f"Output path {OUTPUT_PATH} already exists. Please remove it before running the script."
            )
            return

        # Load IDs and sequences from CSV
        self._load_in_csv()

        # Search for IDs in XML
        self.all_list = self._search_xml_for_ids()

        # Extend found entries with sequences and save to CSV
        self.dataframe_found_entries(OUTPUT_PATH)

    def _load_in_csv(self):
        """
        Load data from the uniprot CSV file.
        It searches for the 'ID' and 'Sequences' columns and stores them in sets for fast lookup.
        """
        # Check if input CSV exists
        if not os.path.exists(self.input_path_csv):
            raise FileNotFoundError(f"CSV file not found: {self.input_path_csv}")

        # Load CSV
        df = pd.read_csv(self.input_path_csv)

        # Drop 'categories' column if it exists
        if "categories" in df.columns:
            df = df.drop("categories")
        # or proceed normally
        else:
            print("No categories column found, proceeding...")

        # Extract IDs and sequences
        list_ids = df[ID_COLUMN].to_list()
        list_seq = df[SEQUENCE_COLUMN].to_list()

        # Strip pipe '|' characters from IDs if present to get raw uniprot IDs
        stripped_ids = []
        # loop through all ids
        for id_str in list_ids:
            if "|" in id_str:
                parts = id_str.split("|")
                if len(parts) >= 2:
                    stripped_ids.append(
                        parts[1]
                    )  # Take the second part as the actual ID
                else:
                    stripped_ids.append(id_str)  # Keep original if unexpected format
            else:
                stripped_ids.append(id_str)  # Fallback: Keep original if no pipes

        # debugging prints
        # print(f"Example original ID: {list_ids[0] if list_ids else 'None'}")
        # print(f"Example stripped ID: {stripped_ids[0] if stripped_ids else 'None'}")
        # list_ids= ["A0A8B7VBX7", "A0A003", "A0A005", "A0A002", "A0A004", "A0A000",]
        # list_seq= list_seq[:6]  # Limit to first 10 sequences for testing

        # get unique ids and sequences
        self.target_ids = set(list_ids)
        self.target_seqs = set(list_seq)

        # Create a DataFrame for sequence lookup later
        self.compare_frame = pd.DataFrame({"ID": list_ids, "Sequences": list_seq})

        # print len of target ids
        print(f"Loaded {len(self.target_ids)} unique IDs to search for")

        return

    def _search_xml_for_ids(self):
        """
        Search the large XML file for target IDs using streaming parser.
        It takes a lot of ram to to handle the large XML file and uniprot csv. Current recommendation is to have at least 2 tb of ram. Not tested with less.
        It extracts relevant 'lcn' data for matches starting with 'PF' for each target protein ID.
        Return:
            self.all_lists: A list of dictionaries containing found entries with keys: 'id', 'taxid', 'lcn_data'

        """
        # Check if input XML exists
        if not os.path.exists(self.input_path_xml):
            raise FileNotFoundError(f"XML file not found: {self.input_path_xml}")

        # Initialize variables for parsing
        self.all_lists = []
        self.processed = 0
        self.failed = []
        self.t0 = time.time()
        # set up outer_self for handler access
        outer_self = self

        class MyHandler(xml.sax.ContentHandler):
            """
            XML parser handler to extract relevant protein and domain data.
            It looks for 'protein' elements with IDs in target_ids and extracts 'lcn' data from 'match' elements starting with 'PF'.
            The parser streams through the XML file to handle large file sizes efficiently.
            It is split into startElement and endElement methods for handling XML tags.
            At the end the handler is called to parse the XML file.
            Returns:
                A list of dictionaries containing found entries with keys: 'id', 'taxid', 'lcn_data'
            """

            def __init__(self, outer_self):
                # initialize parent class and variables
                self.outer_self = outer_self
                self.current_protein_id = None
                self.current_match_id = None
                self.current_taxid = None
                self.current_match_name = None
                self.in_protein = False
                self.in_match = False
                self.current_lcn_data = []

            def startElement(self, name, attrs):
                """
                Start element handler for XML parsing.
                It detects 'protein', 'match', and 'lcn' elements and extracts relevant attributes
                Args:
                    name (str): Name of the XML element.
                    attrs (xml.sax.xmlreader.AttributesImpl): Attributes of the XML element.
                """
                # get protein id and taxid and check if in target ids
                if name == "protein":
                    # get protein id and taxid
                    protein_id = attrs.get("id", "")
                    taxid = attrs.get("taxid", "")
                    # check if protein id is in target ids
                    if protein_id in self.outer_self.target_ids:
                        # update variables
                        self.current_protein_id = protein_id
                        self.current_taxid = taxid
                        self.in_protein = True
                        self.current_lcn_data = []  # reset lcn data for new protein

                # check for match elements within protein
                elif name == "match" and self.in_protein:
                    # get match id and check if starts with 'PF'
                    match_id = attrs.get("id", "")
                    # if pfam id found
                    if match_id.startswith("PF"):
                        # update variables
                        self.in_match = True
                        self.current_match_id = match_id
                        self.current_match_name = attrs.get("name", "")

                # check for lcn elements within match
                elif name == "lcn" and self.in_match:
                    # get domain start and end if available
                    if "start" in attrs and "end" in attrs:
                        # append lcn data to current dictionary list
                        self.current_lcn_data.append(
                            {
                                "start": int(attrs["start"]),
                                "end": int(attrs["end"]),
                                "fragments": attrs.get("fragments", ""),
                                "score": attrs.get("score", ""),
                                "match_id": self.current_match_id,
                                "match_name": self.current_match_name,
                            }
                        )

            def endElement(self, name):
                """
                End element handler for XML parsing.
                It finalizes data extraction for 'protein' and 'match' elements.
                """
                # check for end of protein element
                if name == "protein" and self.in_protein:
                    # finalize and store protein data if any lcn data found
                    if self.current_lcn_data:
                        # Extract only start, end, and match_id from lcn_data
                        # therefore simplify the lcn_data entries
                        simplified_lcn_data = []
                        for lcn in self.current_lcn_data:
                            # get only start, end, match_id(domain ID)
                            simplified_lcn_data.append(
                                {
                                    "start": lcn["start"],
                                    "end": lcn["end"],
                                    "match_id": lcn["match_id"],
                                }
                            )
                        # update current_lcn_data
                        self.current_lcn_data = simplified_lcn_data

                        # store protein data
                        protein_data = {
                            "id": self.current_protein_id,
                            "taxid": self.current_taxid,
                            "lcn_data": self.current_lcn_data,
                        }
                        # get all lists from outer self
                        self.outer_self.all_lists.append(protein_data)
                    # if no lcn data found, skip protein
                    else:
                        pass

                    # Increment progress counter
                    self.outer_self.processed += 1

                    # Print progress every 10000 processed proteins
                    if self.outer_self.processed % 10000 == 0:
                        # prints time and rate, with found targets and failed targets and entries collected
                        elapsed_time = time.time() - self.outer_self.t0
                        rate = (
                            self.outer_self.processed / elapsed_time
                            if elapsed_time > 0
                            else 0
                        )
                        print(
                            f"Targets found: {self.outer_self.processed}/{len(self.outer_self.target_ids)} | "
                            f"Failed: {len(self.outer_self.failed)} | "
                            f"Entries collected: {len(self.outer_self.all_lists)} | "
                            f"Rate: {rate:.2f} targets/sec | "
                            f"Elapsed: {elapsed_time:.1f}s",
                            end="\r",
                            flush=True,
                        )

                    # reset variables for next protein
                    self.in_protein = False
                    self.current_protein_id = None
                    self.current_taxid = None
                    self.current_lcn_data = []

                # check for end of match element, reset match variables
                elif name == "match":
                    self.in_match = False
                    self.current_match_id = None
                    self.current_match_name = None

        # Create XML parser and parse the file
        handler = MyHandler(outer_self)
        xml.sax.parse(self.input_path_xml, handler)

        return self.all_lists

    def dataframe_found_entries(self, output_path):
        """
        Pooles found XML entries into a pandas DataFrame with relevant columns.
        """

        # Check if any entries were found, otherwise abort
        if not self.all_list:
            print("Warning: No entries found to save.")
            return

        # Create DataFrame from found entries with relevant 4 columns
        try:
            self.df = pd.DataFrame(
                self.all_list, columns=["start", "end", "id", "Pfam_id"]
            )
            print(f"Created DataFrame with {len(self.df)} entries")

            # check if csv file exists
            if not os.path.exists(self.input_path_csv):
                raise FileNotFoundError(f"CSV file not found: {self.input_path_csv}")

            # Create a dictionary for O(1) lookup
            sequence_dict = {}
            # loop through compare_frame to fill dictionary
            for _, row in self.compare_frame.iterrows():
                entry_id = row["ID"]
                sequence = row["Sequences"]
                sequence_dict[entry_id] = sequence

            # Flatten the nested structure and extend with sequences
            flattened_entries = []
            for entry in self.all_list:
                entry_id = entry["id"]  # get protein id
                taxid = entry["taxid"]  # get taxid
                sequence = sequence_dict.get(
                    entry_id, None
                )  # get sequence from dictionary
                # Create rows for each lcn_data entry
                # get all lcn data
                for lcn in entry["lcn_data"]:
                    # append new entry with sequence
                    flattened_entries.append(
                        [
                            lcn["start"],  # domain start
                            lcn["end"],  # domain end
                            entry_id,  # protein id
                            lcn["match_id"],  # domain pfam id
                            taxid,  # taxid
                            sequence,  # protein sequence
                        ]
                    )

            # Convert to DataFrame and save to CSV
            if flattened_entries:
                self.df = pd.DataFrame(
                    flattened_entries,
                    columns=["start", "end", "id", "Pfam_id", "taxid", "Sequence"],
                )
                print(f"Extended entries with sequences, total entries: {len(self.df)}")
                self.df.to_csv(output_path, index=False)
                print(
                    f"Successfully saved {len(flattened_entries)} entries to {output_path}"
                )
            # if no entries to save abort
            else:
                print("No entries to save after flattening.")
                raise ValueError("No entries to save after flattening.")

        # handle exceptions if something goes wrong
        except Exception as e:
            print(f"Error in extend_with_seq: {e}")
            import traceback

            traceback.print_exc()
            raise


#############################################################################################
# call init to start preprocessing
DatasetPreprocessor(
    input_path_xml=XML_PATH,
    input_path_csv=CSV_PATH,
)
