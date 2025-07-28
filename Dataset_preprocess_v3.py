import os
import xml.sax
import pandas as pd
import time

XML_PATH = "/global/scratch2/sapelt/Protein_matched_complete/Protein_match_complete.xml"
CSV_PATH = "/global/scratch2/sapelt/Protein_matched_complete/uniprot_full.csv"
OUTPUT_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesCompleteProteins_tax.csv"



class DatasetPreprocessor:
    def __init__(self, input_path_xml, input_path_csv):
        self.input_path_xml = input_path_xml
        self.input_path_csv = input_path_csv

        print("Starting dataset preprocessing...")

        if os.path.exists(OUTPUT_PATH):
            print(f"Output path {OUTPUT_PATH} already exists. Please remove it before running the script.")
            return

        list_ids, list_seq = self._load_in_csv()
        
        # testset:
        # list_ids= ["A0A8B7VBX7", "A0A003", "A0A005", "A0A002", "A0A004", "A0A000",]
        # list_seq= list_seq[:6]  # Limit to first 10 sequences for testing

        self.target_ids = set(list_ids)
        self.target_seqs = set(list_seq)


        self.compare_frame= pd.DataFrame(
            {
                "ID": list_ids,
                "Sequences": list_seq
            }
        )

        # print(self.compare_frame.head(10))  # Display first 10 rows for verification


        print(f"Loaded {len(self.target_ids)} unique IDs to search for")
        # print(list_ids)

        # Search for IDs in XML
        self.all_list = self._search_xml_for_ids()


        # Save found entries to a file
        self.save_found_entries()

        self.extend_with_seq(OUTPUT_PATH)

    def _load_in_csv(self):
        """Load data from a CSV file."""
        if not os.path.exists(self.input_path_csv):
            raise FileNotFoundError(f"CSV file not found: {self.input_path_csv}")

        df = pd.read_csv(self.input_path_csv)

        if "categories" in df.columns:
            df = df.drop("categories")
        else:
            print("No categories column found, proceeding...")

        list_ids = df["ID"].to_list()
        list_seq = df["Sequences"].to_list()
        
        stripped_ids = []
        for id_str in list_ids:
            if '|' in id_str:
                parts = id_str.split('|')
                if len(parts) >= 2:
                    stripped_ids.append(parts[1])
                else:
                    stripped_ids.append(id_str)  # Keep original if unexpected format
            else:
                stripped_ids.append(id_str)  # Fallback: Keep original if no pipes
        
        # print(f"Example original ID: {list_ids[0] if list_ids else 'None'}")
        # print(f"Example stripped ID: {stripped_ids[0] if stripped_ids else 'None'}")
        
        return stripped_ids, list_seq

    def _search_xml_for_ids(self):
        """Search the large XML file for target IDs using streaming parser."""
        if not os.path.exists(self.input_path_xml):
            raise FileNotFoundError(f"XML file not found: {self.input_path_xml}")

        found_entries = {}
        self.all_lists     = []
        self.processed     = 0
        self.failed        = []
        self.t0 = time.time()

        # Add progress printing function
        def print_progress():
            elapsed_time = time.time() - self.t0
            rate = self.processed / elapsed_time if elapsed_time > 0 else 0
            print(f"Progress: {self.processed} processed, {len(self.failed)} failed, "
                f"{len(self.all_lists)} found, "
                f"Rate: {rate:.2f} entries/sec, "
                f"Elapsed: {elapsed_time:.1f}s")

        outer_self = self


        class MyHandler(xml.sax.ContentHandler):
            def __init__(self, outer_self):
                self.outer_self = outer_self
                self.current_protein_id = None
                self.current_match_id = None
                self.current_taxid = None
                self.current_match_name = None
                self.in_protein = False
                self.in_match = False
                self.current_lcn_data = []
                
            def startElement(self, name, attrs):
                if name == 'protein':
                    protein_id = attrs.get('id', '')
                    taxid = attrs.get('taxid', '')
                    if protein_id in self.outer_self.target_ids:
                        # print(f"Found target protein ID: {protein_id}")
                        self.current_protein_id = protein_id
                        self.current_taxid = taxid 
                        self.in_protein = True
                        self.current_lcn_data = []
                        
                elif name == 'match' and self.in_protein:
                    match_id = attrs.get('id', '')
                    if match_id.startswith('PF'):
                        self.in_match = True
                        self.current_match_id = match_id
                        self.current_match_name = attrs.get('name', '')
                        
                elif name == 'lcn' and self.in_match:
                    if 'start' in attrs and 'end' in attrs:
                        self.current_lcn_data.append({
                            'start': int(attrs['start']),
                            'end': int(attrs['end']),
                            'fragments': attrs.get('fragments', ''),
                            'score': attrs.get('score', ''),
                            'match_id': self.current_match_id,
                            'match_name': self.current_match_name
                        })
            
            def endElement(self, name):
                if name == 'protein' and self.in_protein:
                    if self.current_lcn_data:
                        # Extract only start, end, and match_id from lcn_data
                        simplified_lcn_data = []
                        for lcn in self.current_lcn_data:
                            simplified_lcn_data.append({
                                'start': lcn['start'],
                                'end': lcn['end'],
                                'match_id': lcn['match_id']
                            })
                        self.current_lcn_data = simplified_lcn_data

                        protein_data = {
                            'id': self.current_protein_id,
                            'taxid': self.current_taxid,
                            'lcn_data': self.current_lcn_data
                        }
                        self.outer_self.all_lists.append(protein_data)
                        # print(f"Added data for protein ID: {self.current_protein_id}")
                        # print(f"LCN data: {self.current_lcn_data}")
                    else:
                        # print(f"No LCN data found for protein ID: {self.current_protein_id}")
                        pass
                    
                    # Increment progress counter
                    self.outer_self.processed += 1
                    
                    # Print progress every 1 processed proteins
                    if self.outer_self.processed % 10000 == 0:
                        elapsed_time = time.time() - self.outer_self.t0
                        rate = self.outer_self.processed / elapsed_time if elapsed_time > 0 else 0
                        print(f"Targets found: {self.outer_self.processed}/{len(self.outer_self.target_ids)} | "
                              f"Failed: {len(self.outer_self.failed)} | "
                              f"Entries collected: {len(self.outer_self.all_lists)} | "
                              f"Rate: {rate:.2f} targets/sec | "
                              f"Elapsed: {elapsed_time:.1f}s", end='\r', flush=True)
                    
                    self.in_protein = False
                    self.current_protein_id = None
                    self.current_taxid = None
                    self.current_lcn_data = []
                    
                elif name == 'match':
                    self.in_match = False
                    self.current_match_id = None
                    self.current_match_name = None

        handler = MyHandler(outer_self)
        xml.sax.parse(self.input_path_xml, handler)

        return self.all_lists

    def _extract_protein_data(self, elem):
        """Extract ID and lcn data from protein element."""
        
        protein_id = elem.get('id', '')
        print(protein_id)
        lcn_data = []
        
        # Find all match elements
        match_elements = elem.findall('.//match')
        # print(match_elements)
        # print(f"Found {len(match_elements)} match elements")
    
        # Find all lcn elements within this protein
        for i, match_elem in enumerate(match_elements):
            # print(f"Match {i}: tag={match_elem.tag}")
            # print(f"  Attributes dict: {dict(match_elem.attrib)}")
            # print(f"  id={match_elem.attrib.get('id', 'NO_ID')}, name={match_elem.attrib.get('name', 'NO_NAME')}")
            # print(match_elem.attrib.get('id'))
            id_match = match_elem.attrib.get('id')
            
            if id_match.startswith('PF'):
                # print(f"  Match ID starts with 'PF': {id_match}")


                lcn_elements = match_elem.findall('lcn')
                # print(f"  Found {len(lcn_elements)} lcn elements in this match")
                # print(lcn_elements.attrib)
                for j, lcn_elem in enumerate(lcn_elements):
                    # print(f"    LCN {j}: attributes={dict(lcn_elem.attrib)}")
                    if 'start' in lcn_elem.attrib and 'end' in lcn_elem.attrib:
                        lcn_data.append({
                            'start': int(lcn_elem.attrib['start']),
                            'end': int(lcn_elem.attrib['end']),
                            'fragments': lcn_elem.attrib.get('fragments', ''),
                            'score': lcn_elem.attrib.get('score', ''),
                            'match_id': match_elem.attrib.get('id', ''),
                            'match_name': match_elem.attrib.get('name', '')
                        })
                        # print(f"    Added LCN data for match_id: {match_elem.attrib.get('id', '')}")

                for lcn_elem in lcn_elements:
                    lcn_elem.clear()  # Clear lcn elements to free memory
                

            # Clear match
            match_elem.clear()

        # print(f"Total lcn_data collected: {len(lcn_data)}")
        return {
            'id': protein_id,
            'lcn_data': lcn_data
        }

    def save_found_entries(self):
        """Pooles found XML entries."""
        if not self.all_list:
            print("Warning: No entries found to save.")
            return
        
        try:
            # # Ensure output directory exists
            # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            
            # Create a DataFrame from the found entries
            self.df = pd.DataFrame(self.all_list, columns=["start", "end", "id", "Pfam_id"])
            print(f"Created DataFrame with {len(self.df)} entries")
            
        except Exception as e:
            print(f"Error in save_found_entries: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extend_with_seq(self, output_path):
        """Extend the found entries with sequences from the CSV file."""
        try:
            if not os.path.exists(self.input_path_csv):
                raise FileNotFoundError(f"CSV file not found: {self.input_path_csv}")

            # Create a dictionary for O(1) lookup
            sequence_dict = {}
            for _, row in self.compare_frame.iterrows():
                entry_id = row["ID"]  
                sequence = row["Sequences"]
                # print(entry_id, sequence)  
                sequence_dict[entry_id] = sequence

            # print(sequence_dict)

            # Flatten the nested structure and extend with sequences
            flattened_entries = []
            # print(self.all_list)
            for entry in self.all_list:
                entry_id = entry['id']  # Use dictionary key instead of index
                taxid = entry['taxid']
                sequence = sequence_dict.get(entry_id, None)
                # print(sequence)
                # Create rows for each lcn_data entry
                for lcn in entry['lcn_data']:
                    flattened_entries.append([
                        lcn['start'],
                        lcn['end'], 
                        entry_id,
                        lcn['match_id'],
                        taxid,
                        sequence
                    ])

            # Convert to DataFrame
            if flattened_entries:
                self.df = pd.DataFrame(flattened_entries, columns=["start", "end", "id", "Pfam_id", "taxid","Sequence"])
                print(f"Extended entries with sequences, total entries: {len(self.df)}")

                # Save to CSV
                self.df.to_csv(output_path,index=False)
                print(f"Successfully saved {len(flattened_entries)} entries to {output_path}")
            else:
                print("No entries to save after flattening.")
                
        except Exception as e:
            print(f"Error in extend_with_seq: {e}")
            import traceback
            traceback.print_exc()
            raise
#############################################################################################
DatasetPreprocessor(
    input_path_xml=XML_PATH,
    input_path_csv=CSV_PATH,
)
