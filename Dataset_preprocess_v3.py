import os
import lxml.etree as ET
import polars as pl
import psutil


XML_PATH = "/global/scratch2/sapelt/Protein_matched_complete/Protein_match_complete.xml"
CSV_PATH = "/global/scratch2/sapelt/Protein_matched_complete/uniprot_sprot.csv"
OUTPUT_PATH = "./FoundEntries.csv"



class DatasetPreprocessor:
    def __init__(self, input_path_xml, input_path_csv):
        self.input_path_xml = input_path_xml
        self.input_path_csv = input_path_csv

        list_ids, list_seq = self._load_in_csv()
        
        # testset:
        # list_ids= ["A0A001", "A0A003", "A0A005", "A0A002", "A0A004", "A0A000",]
        # list_seq= list_seq[:6]  # Limit to first 10 sequences for testing

        self.target_ids = set(list_ids)
        self.target_seqs = set(list_seq)


        self.compare_frame= pl.DataFrame(
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

        df = pl.read_csv(self.input_path_csv)

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
                stripped_ids.append(id_str)  # Keep original if no pipes
        
        # print(f"Example original ID: {list_ids[0] if list_ids else 'None'}")
        # print(f"Example stripped ID: {stripped_ids[0] if stripped_ids else 'None'}")
        
        return stripped_ids, list_seq

    def _search_xml_for_ids(self):
        """Search the large XML file for target IDs using streaming parser."""
        if not os.path.exists(self.input_path_xml):
            raise FileNotFoundError(f"XML file not found: {self.input_path_xml}")
        
        found_entries = {}
        processed_count = 0
        all_elements_count = 0
        
        # Add timing variables for ETA calculation
        import time
        start_time = time.time()
        last_update_time = start_time
        
        # Use iterparse for memory-efficient streaming
        context = ET.iterparse(self.input_path_xml, events=('start', 'end'))
        context = iter(context)
        
        all_lists = []

        try:
            event, root = next(context)
            print(f"Root element: {root.tag}")  # Add this back for debugging
            
            for event, elem in context:
                all_elements_count += 1
                
                if event == 'end':
                    # Look for protein elements based on your XML structure
                    if elem.tag == 'protein':
                        
                        # Extract ID and lcn data from protein element
                        entry_data = self._extract_protein_data(elem)
                        
                        if entry_data:
                            if entry_data['id'] in self.target_ids:
                                found_entries[entry_data['id']] = {
                                    'xml_content': ET.tostring(elem, encoding='unicode'),
                                    'lcn_data': entry_data['lcn_data']
                                }

                                if entry_data['lcn_data']:
                                    for lcn in entry_data['lcn_data']:
                                        if lcn['match_id'].startswith('PF'):
                                            templist = [lcn['start'], lcn['end'], entry_data['id'], lcn['match_id']]
                                            all_lists.append(templist)

                        processed_count += 1
                        
                        # IMPORTANT: Clear the element to free memory
                        elem.clear()
                        while elem.getprevious() is not None:
                            del elem.getparent()[0]
                        
                        # Enhanced progress reporting
                        if processed_count % 10000 == 0:
                            current_time = time.time()
                            elapsed_time = current_time - start_time
                            
                            if processed_count > 0:
                                rate = processed_count / elapsed_time
                                print(f"Processed {processed_count} entries | "
                                      f"Found {len(found_entries)}/{len(self.target_ids)} targets | "
                                      f"Pfam entries: {len(all_lists)} | "
                                      f"Rate: {rate:.1f} entries/s | ", end='', flush=True)
                            
                            process = psutil.Process(os.getpid())
                            mem_mb = process.memory_info().rss / 1024 / 1024
                            print(f"Memory usage: {mem_mb:.2f} MB", end='\r', flush=True)

                # Check if we found all targets
                if len(found_entries) == len(self.target_ids):
                    final_time = time.time()
                    total_elapsed = final_time - start_time
                    print(f"\nFound all target IDs! "
                        f"Processed {processed_count} entries in {total_elapsed:.2f}s "
                        f"({total_elapsed/60:.2f}min) | "
                        f"Pfam entries: {len(all_lists)}")
                    break

        except ET.XMLSyntaxError as e:
            print(f"XML parsing error: {e}")
            raise  # Re-raise to see the full error
        except MemoryError as e:
            print(f"Memory error: {e}")
            print(f"Processed {processed_count} entries before memory error")
            raise
        except Exception as e:
            print(f"Unexpected error during XML processing: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Final summary
        final_time = time.time()
        total_elapsed = final_time - start_time
        
        print(f"\nSearch complete!")
        print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
        print(f"Found {len(found_entries)} out of {len(self.target_ids)} target IDs")
        print(f"Processed {processed_count} protein entries")
        print(f"Found {len(all_lists)} Pfam domain entries")
        
        return all_lists

    def _extract_protein_data(self, elem):
        """Extract ID and lcn data from protein element."""
        
        protein_id = elem.attrib['id']
        lcn_data = []
        
        # Find all lcn elements within this protein
        for match_elem in elem.findall('.//match'):
            for lcn_elem in match_elem.findall('lcn'):
                if 'start' in lcn_elem.attrib and 'end' in lcn_elem.attrib:
                    lcn_data.append({
                        'start': int(lcn_elem.attrib['start']),  # Convert to int
                        'end': int(lcn_elem.attrib['end']),      # Convert to int
                        'fragments': lcn_elem.attrib.get('fragments', ''),
                        'score': lcn_elem.attrib.get('score', ''),
                        'match_id': match_elem.attrib.get('id', ''),  # Add match ID for reference
                        'match_name': match_elem.attrib.get('name', '')  # Add match name
                    })

        return {
            'id': protein_id,
            'lcn_data': lcn_data
        }

    def save_found_entries(self):
        """Save found XML entries to a csv."""
        if not self.all_list:
            print("Warning: No entries found to save.")
            return
        
        try:
            # # Ensure output directory exists
            # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            
            # Create a DataFrame from the found entries
            self.df = pl.DataFrame(self.all_list, schema=["start", "end", "id", "Pfam_id"])
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

            # Ensure output directory exists
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create a dictionary for O(1) lookup
            sequence_dict = {}
            for row in self.compare_frame.iter_rows():
                entry_id = row[0]  
                sequence = row[1]  
                sequence_dict[entry_id] = sequence

            # Extend entries with sequences
            for entry in self.all_list:
                entry_id = entry[2]
                if entry_id in sequence_dict:
                    entry.append(sequence_dict[entry_id])
                else:
                    entry.append(None)

            # Convert to DataFrame
            self.df = pl.DataFrame(self.all_list, schema=["start", "end", "id", "Pfam_id", "Sequence"])
            print(f"Extended entries with sequences, total entries: {len(self.df)}")

            # Save to CSV
            self.df.write_csv(output_path)
            print(f"Successfully saved {len(self.all_list)} entries to {output_path}")
            
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
