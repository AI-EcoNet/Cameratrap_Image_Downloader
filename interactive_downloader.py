import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import concurrent.futures
import re
import pandas as pd
import threading

# Get the folder where the current script is located
BASE_DIR = Path(__file__).resolve().parent

# Define paths relative to the script location
CATEGORIES_FILE = BASE_DIR / r"DataFiles\categories.json"
CATALOG_FILE = BASE_DIR / r"DataFiles\animal_catalog.parquet"
METADATA_FILE = BASE_DIR / r"DataFiles\catalog_metadata.json"
OUTPUT_ROOT = BASE_DIR / "Downloads"

# Networking settings
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 180
MAX_RETRIES = 5
MAX_WORKERS = 8
MIN_VALID_BYTES = 5 * 1024
STATUS_FORCELIST = (429, 500, 502, 503, 504)
BACKOFF_BASE = 0.75
MAX_BACKOFF = 30.0
VALIDATE_WITH_PIL = True

class NavigationException(Exception):
    """Custom exception for navigation commands"""
    def __init__(self, action):
        self.action = action
        super().__init__(action)

def show_loading_animation(message, stop_event):
    """Show a loading animation"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop_event.is_set():
        print(f"\r{chars[i % len(chars)]} {message}", end='', flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r✓ {message} - Complete!", flush=True)

def load_categories():
    """Load the categories taxonomy"""
    print("📂 Loading categories taxonomy...")
    try:
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except KeyboardInterrupt:
        print("\n❌ Loading cancelled by user.")
        sys.exit(1)

def load_catalog():
    """Load the optimized catalog (Parquet + metadata)"""
    
    stop_event = threading.Event()
    loading_thread = threading.Thread(
        target=show_loading_animation, 
        args=("Loading catalog (this may take a moment)", stop_event)
    )
    loading_thread.start()
    
    try:
        # Load metadata first
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load main catalog DataFrame
        df = pd.read_parquet(CATALOG_FILE)
        
        stop_event.set()
        loading_thread.join()
        
        return {"data": df, "metadata": metadata}
    except KeyboardInterrupt:
        stop_event.set()
        loading_thread.join()
        print("\n❌ Loading cancelled by user.")
        sys.exit(1)

def parse_selection(input_str, max_val):
    """Parse user selection string with navigation support"""
    input_str = input_str.strip().lower()
    
    # Check for navigation commands first
    if input_str in ['b', 'back']:
        raise NavigationException('back')
    elif input_str in ['r', 'restart']:
        raise NavigationException('restart')
    elif input_str in ['q', 'quit']:
        raise NavigationException('quit')
    
    # Handle empty input - return None to indicate no selection
    if not input_str:
        return None
    
    # Parse numbers
    numbers = re.split(r'[,\s]+', input_str.strip())
    
    selected = []
    valid_input = False
    
    for num_str in numbers:
        if num_str:  # Skip empty strings
            try:
                num = int(num_str)
                if 1 <= num <= max_val:
                    selected.append(num - 1)
                    valid_input = True
                else:
                    print(f"⚠️  {num} is out of range (1-{max_val})")
            except ValueError:
                print(f"⚠️  '{num_str}' is not a valid number")
    
    if not valid_input:
        return []  # Invalid input, but not navigation
    
    return list(set(selected))

def consolidate_similar_species(species_counts):
    """Consolidate similar species by Latin name properly"""
    # Group by Latin name first
    latin_groups = defaultdict(list)
    
    for species_name, count in species_counts:
        # Extract Latin name (part in parentheses)
        match = re.search(r'\(([^)]+)\)', species_name)
        if match:
            latin_name = match.group(1).strip()
            latin_groups[latin_name].append((species_name, count))
        else:
            # No Latin name, keep as is
            latin_groups[species_name].append((species_name, count))
    
    consolidated = []
    
    for latin_name, species_list in latin_groups.items():
        if len(species_list) == 1:
            # Only one species with this Latin name
            consolidated.append(species_list[0])
        else:
            # Multiple species with same Latin name - consolidate
            total_count = sum(count for _, count in species_list)
            
            # Choose the most general name (shortest common part before parentheses)
            species_names = [name for name, _ in species_list]
            common_parts = []
            
            for name in species_names:
                if '(' in name:
                    common_part = name.split('(')[0].strip()
                    common_parts.append(common_part)
                else:
                    common_parts.append(name)
            
            # Find the shortest common part as it's usually more general
            best_common = min(common_parts, key=len)
            
            # Create consolidated name with Latin name
            if '(' in species_names[0]:  # Has Latin name
                final_name = f"{best_common} ({latin_name})"
            else:
                final_name = best_common
            
            consolidated.append((final_name, total_count))
    
    # Sort by count descending
    consolidated.sort(key=lambda x: x[1], reverse=True)
    return consolidated

def display_main_menu(categories, df):
    """Display main category menu with actual counts"""
    print("\n" + "="*70)
    print("🦁 ANIMAL DATASET INTERACTIVE DOWNLOADER")
    print("="*70)
    
    main_classes = [c for c in categories.keys() if c != "Other / Non-wildlife"]
    
    # Get actual counts from DataFrame
    available_classes = []
    for class_name in main_classes:
        count = df[df['animal_class'] == class_name].shape[0]
        if count > 0:
            available_classes.append((class_name, count))
    
    if not available_classes:
        print("❌ No classes with available images found!")
        return []
    
    # Sort by count descending
    available_classes.sort(key=lambda x: x[1], reverse=True)
    
    print("🐾 Available animal classes:")
    print()
    
    for i, (class_name, count) in enumerate(available_classes, 1):
        print(f"  {i:2d}. {class_name:<15} ({count:,} images)")
    
    print("\n" + "-"*50)
    print("💡 Navigation options:")
    print("  • Enter numbers: '1 3 5' or '1,2,3'")
    print("  • Press Enter: select all classes")
    print("  • Type 'q': quit program")
    print("-"*50)
    
    return available_classes

def display_family_menu(class_name, df, family_species_mapping):
    """Display family/group menu for a class using optimized counting"""
    print(f"\n--- 📁 {class_name} ---")
    
    # Get all family mappings for this class
    class_families = {}
    for mapping_key, species_list in family_species_mapping.items():
        if mapping_key.startswith(f"{class_name}|"):
            family_name = mapping_key.split("|", 1)[1]
            class_families[family_name] = species_list
    
    if not class_families:
        print(f"❌ No family mappings found for {class_name}")
        return []
    
    print("📂 Available families/groups:")
    print()
    
    # Get species counts for this class in one operation
    class_species_counts = df[df['animal_class'] == class_name]['animal_species'].value_counts().to_dict()
    
    # Calculate family counts efficiently
    available_families = []
    for family_name, species_list in class_families.items():
        count = sum(class_species_counts.get(species, 0) for species in species_list)
        if count > 0:
            available_families.append((family_name, count, species_list))
    
    # Sort and display
    available_families.sort(key=lambda x: x[1], reverse=True)
    
    for i, (family_name, count, _) in enumerate(available_families, 1):
        print(f"  {i:2d}. {family_name:<25} ({count:,} images)")
    
    print("\n" + "-"*50)
    print("💡 Navigation:")
    print("  • Enter numbers or press Enter for all")
    print("  • 'b': back to classes")
    print("  • 'r': restart • 'q': quit")
    print("-"*50)
    
    return available_families

def display_species_menu(family_name, species_list, df, class_name, max_display=50):
    """Display species menu for a family with actual counts and consolidation"""
    print(f"\n--- 🔬 {family_name} ---")
    
    if not species_list:
        print(f"❌ No species found for {family_name}")
        return []
    
    
    # Get actual counts for species in this family
    species_counts = []
    for species_name in species_list:
        count = df[
            (df['animal_class'] == class_name) & 
            (df['animal_species'] == species_name)
        ].shape[0]
        
        if count > 0:  # Only include species with images
            species_counts.append((species_name, count))
    
    # DEBUG: Also check for any bird-related species that might be missing
    if family_name == "Unspecified Family" and class_name == "Aves":
        print("🔍 Searching for additional bird species in this class...")
        all_bird_species = df[df['animal_class'] == class_name]['animal_species'].value_counts()
        bird_keywords = ['bird', 'prey', 'raptor', 'owl', 'songbird']
        
        for species_name, count in all_bird_species.items():
            if any(keyword in species_name.lower() for keyword in bird_keywords):
                # Check if this species is already in our list
                if species_name not in [s[0] for s in species_counts]:
                    print(f"    🎯 Found additional species: {species_name} ({count:,} images)")
                    species_counts.append((species_name, count))
    
    if not species_counts:
        print(f"❌ No species with images found for {family_name}")
        return []
    
    # Consolidate similar species
    consolidated_species = consolidate_similar_species(species_counts)
    
    # Create mapping from consolidated names back to original species
    consolidated_to_original = {}
    for consolidated_name, _ in consolidated_species:
        # Find all original species that map to this consolidated name
        original_species = []
        consolidated_latin = re.search(r'\(([^)]+)\)', consolidated_name)
        
        if consolidated_latin:
            consolidated_latin = consolidated_latin.group(1)
            for orig_name, orig_count in species_counts:
                orig_latin = re.search(r'\(([^)]+)\)', orig_name)
                if orig_latin and orig_latin.group(1) == consolidated_latin:
                    original_species.append(orig_name)
        else:
            # No Latin name, direct match
            for orig_name, orig_count in species_counts:
                if orig_name == consolidated_name:
                    original_species.append(orig_name)
        
        consolidated_to_original[consolidated_name] = original_species
    
    total_species = len(consolidated_species)
    display_species = consolidated_species[:max_display]
    
    print(f"🐾 Available species ({len(display_species)} of {total_species}):")
    if total_species > max_display:
        print(f"    (Showing top {max_display} by image count)")
    print()
    
    for i, (species_name, count) in enumerate(display_species, 1):
        print(f"  {i:2d}. {species_name:<45} ({count:,} images)")
    
    if total_species > max_display:
        remaining_count = sum(count for _, count in consolidated_species[max_display:])
        print(f"\n⚠️  Note: {total_species - max_display} more species available ({remaining_count:,} images)")
        print("    Press Enter to select all species in this family")
    
    print("\n" + "-"*60)
    print("💡 Navigation:")
    print("  • Enter numbers or press Enter for all")
    print("  • 'b': back to families")
    print("  • 'r': restart • 'q': quit")
    print("-"*60)
    
    return display_species, consolidated_species, consolidated_to_original

def debug_family_mapping(class_name, family_name, family_species_mapping, df):
    """Debug function to show what species should be in a family"""
    mapping_key = f"{class_name}|{family_name}"
    
    if mapping_key in family_species_mapping:
        expected_species = family_species_mapping[mapping_key]
        print(f"\n🔍 DEBUG: {family_name} should contain:")
        for species in expected_species[:10]:  # Show first 10
            count = df[(df['animal_class'] == class_name) & (df['animal_species'] == species)].shape[0]
            print(f"    {species}: {count:,} images")
        
        if len(expected_species) > 10:
            print(f"    ... and {len(expected_species) - 10} more")
    else:
        print(f"\n🔍 DEBUG: No mapping found for {mapping_key}")
        
        # Search for similar keys
        similar_keys = [k for k in family_species_mapping.keys() if family_name.lower() in k.lower()]
        if similar_keys:
            print(f"    Similar keys found: {similar_keys[:3]}")

def get_user_input(prompt=">>> "):
    """Get user input with proper handling"""
    while True:
        try:
            user_input = input(prompt).strip()
            return user_input
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user.")
            sys.exit(1)

def collect_user_selections(categories, catalog):
    """Collect user selections through interactive menu with navigation"""
    df = catalog["data"]
    family_species_mapping = catalog["metadata"].get("family_species_mapping", {})
    
    while True:  # Main navigation loop
        try:
            # Main class selection
            available_classes = display_main_menu(categories, df)
            if not available_classes:
                return []
            
            while True:
                class_selection = get_user_input()
                
                try:
                    if not class_selection:  # Empty input - select all
                        class_indices = list(range(len(available_classes)))
                        break
                    else:
                        class_indices = parse_selection(class_selection, len(available_classes))
                        if class_indices is None:  # Empty input
                            continue
                        elif class_indices == []:  # Invalid input
                            continue
                        else:
                            break
                except NavigationException as e:
                    if e.action == 'quit':
                        print("👋 Goodbye!")
                        sys.exit(0)
                    elif e.action == 'restart':
                        break
                    else:
                        continue
            
            if not class_indices:
                continue
            
            # Process each selected class
            temp_selections = []
            consolidated_species_count = 0  # Track consolidated count
            
            for class_idx in class_indices:
                class_name, _ = available_classes[class_idx]
                
                while True:  # Family selection loop
                    try:
                        available_families = display_family_menu(class_name, df, family_species_mapping)
                        if not available_families:
                            break
                        
                        while True:
                            family_selection = get_user_input()
                            
                            try:
                                if not family_selection:  # Empty input - select all
                                    family_indices = list(range(len(available_families)))
                                    break
                                else:
                                    family_indices = parse_selection(family_selection, len(available_families))
                                    if family_indices is None:  # Empty input
                                        continue
                                    elif family_indices == []:  # Invalid input
                                        continue
                                    else:
                                        break
                                        
                            except NavigationException as e:
                                if e.action == 'back':
                                    family_indices = None
                                    break
                                elif e.action == 'restart':
                                    temp_selections = []
                                    consolidated_species_count = 0
                                    raise NavigationException('restart')
                                elif e.action == 'quit':
                                    raise NavigationException('quit')
                        
                        if family_indices is None:  # Back was pressed
                            break
                        
                        # Process families
                        for family_idx in family_indices:
                            family_name, family_total, family_species_list = available_families[family_idx]
                            
                            while True:  # Species selection loop
                                try:
                                    result = display_species_menu(family_name, family_species_list, df, class_name)
                                    if not result:
                                        break
                                    
                                    display_species, consolidated_species, consolidated_to_original = result
                                    species_indices = None  # Initialize the variable
                                    
                                    while True:
                                        species_selection = get_user_input()
                                        
                                        try:
                                            if not species_selection:  # Empty input - select all
                                                # Add all original species names from all consolidated groups
                                                for consolidated_name, _ in consolidated_species:
                                                    original_names = consolidated_to_original[consolidated_name]
                                                    for orig_name in original_names:
                                                        temp_selections.append((class_name, orig_name))
                                                # Count consolidated species properly
                                                consolidated_species_count += len(consolidated_species)
                                                species_indices = "all_selected"  # Set to indicate success
                                                break
                                            else:
                                                species_indices = parse_selection(species_selection, len(display_species))
                                                
                                                if species_indices is None:  # Empty input
                                                    continue
                                                elif species_indices == []:  # Invalid input
                                                    continue
                                                else:
                                                    # Select specific consolidated species
                                                    for species_idx in species_indices:
                                                        consolidated_name, _ = display_species[species_idx]
                                                        original_names = consolidated_to_original[consolidated_name]
                                                        for orig_name in original_names:
                                                            temp_selections.append((class_name, orig_name))
                                                    # Count selected consolidated species
                                                    consolidated_species_count += len(species_indices)
                                                    break
                                            
                                        except NavigationException as e:
                                            if e.action == 'back':
                                                species_indices = None
                                                break
                                            elif e.action == 'restart':
                                                raise NavigationException('restart')
                                            elif e.action == 'quit':
                                                raise NavigationException('quit')
                                    
                                    if species_indices is None:  # Back was pressed
                                        break
                                    else:
                                        break  # Successfully selected species
                                        
                                except NavigationException as e:
                                    if e.action == 'restart':
                                        raise
                                    elif e.action == 'quit':
                                        raise
                        
                        break  # Successfully processed families
                        
                    except NavigationException as e:
                        if e.action == 'restart':
                            raise
                        elif e.action == 'quit':
                            raise
            
            # Return selections when we have them
            if temp_selections:
                return temp_selections, consolidated_species_count
            
        except NavigationException as e:
            if e.action == 'quit':
                print("👋 Goodbye!")
                sys.exit(0)
            elif e.action == 'restart':
                continue
        except KeyboardInterrupt:
            print("\n❌ Selection cancelled by user.")
            sys.exit(1)

def analyze_selections(selected_species, catalog):
    """Analyze selections and show statistics with proper timing"""
    print("\n" + "="*70)
    print("📊 DOWNLOAD PREPARATION")
    print("="*70)
    
    stop_event = threading.Event()
    loading_thread = threading.Thread(
        target=show_loading_animation, 
        args=("Analyzing selections and preparing download links", stop_event)
    )
    loading_thread.start()
    
    try:
        df = catalog["data"]
        metadata = catalog["metadata"]
        projects = metadata["projects"]
        
        # Filter DataFrame for selected species
        conditions = []
        for class_name, species_name in selected_species:
            conditions.append(
                (df['animal_class'] == class_name) & 
                (df['animal_species'] == species_name)
            )
        
        if not conditions:
            stop_event.set()
            loading_thread.join()
            return []
        
        # Combine conditions with OR
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition |= condition
        
        filtered_df = df[combined_condition].copy()
        
        if filtered_df.empty:
            stop_event.set()
            loading_thread.join()
            print("❌ No images found for selected species.")
            return []
        
        # Add URL construction - this is the time-consuming part
        filtered_df['url_to_original'] = filtered_df.apply(
            lambda row: f"{projects[row['project_id']]['base_url']}/{row['filename']}" 
            if projects[row['project_id']]['base_url'] else "",
            axis=1
        )
        
        # Convert to list of dicts - also time-consuming for large datasets
        entries = []
        for _, row in filtered_df.iterrows():
            entry = {
                "UUID": row['id'],
                "animal_class": row['animal_class'],
                "animal_species": row['animal_species'],
                "filename": row['filename'],
                "project_name": projects[row['project_id']]['name'],
                "dataset_name": row['dataset'],
                "url_to_original": row['url_to_original'],
                "base_url": projects[row['project_id']]['base_url']
            }
            entries.append(entry)
        
        # Stop loading animation only when everything is ready
        stop_event.set()
        loading_thread.join()
        
        # Print basic statistics
        print(f"🎯 Ready to download: {len(entries):,} images")
        
        return entries
        
    except KeyboardInterrupt:
        stop_event.set()
        loading_thread.join()
        print("\n❌ Analysis cancelled by user.")
        sys.exit(1)

def build_session():
    """Build requests session with proper configuration"""
    session = requests.Session()
    retries = Retry(total=0, backoff_factor=0, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def file_seems_valid(p: Path) -> bool:
    """Check if file exists and is valid"""
    if not p.exists():
        return False
    try:
        size = p.stat().st_size
    except OSError:
        return False
    if size < MIN_VALID_BYTES:
        return False

    if not VALIDATE_WITH_PIL:
        return True

    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def polite_sleep(seconds: float) -> None:
    """Sleep politely"""
    if seconds > 0:
        time.sleep(seconds)

def download_with_validation(url: str, out_path: Path, session: requests.Session) -> Tuple[bool, str]:
    """Robust download function - returns (success, message)"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if file_seems_valid(out_path):
        return True, "exists-valid"

    if out_path.exists():
        try:
            out_path.unlink()
        except OSError:
            return False, "cannot-remove-old-file"

    last_err = "unknown"
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except requests.RequestException as e:
            last_err = f"network {e.__class__.__name__}: {e}"
            backoff = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
            polite_sleep(backoff)
            continue

        status = r.status_code
        if status == 200:
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            try:
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(out_path)
            except Exception as e:
                last_err = f"io-error {e}"
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                backoff = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
                polite_sleep(backoff)
                continue

            if file_seems_valid(out_path):
                return True, "downloaded"
            else:
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                last_err = "validation-failed"
                backoff = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
                polite_sleep(backoff)
                continue

        elif status in STATUS_FORCELIST:
            last_err = f"http {status}"
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_s = float(retry_after)
                except ValueError:
                    wait_s = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
            else:
                wait_s = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
            polite_sleep(wait_s)
            continue
        else:
            reason = r.reason or ""
            last_err = f"http {status} {reason}".strip()
            if 400 <= status < 500:
                polite_sleep(0.05)
                break
            else:
                backoff = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
                polite_sleep(backoff)
                continue

    return False, last_err

def download_images(entries):
    """Download images flat into species folders with metadata"""
    if not entries:
        return
    
    print("\n" + "="*70)
    print("⬇️  DOWNLOAD CONFIRMATION")
    print("="*70)
    
    # Count species
    species_count = len(set(entry["animal_species"] for entry in entries))
    estimated_mb = len(entries) * 0.5
    
    print(f"\n📦 Images to download: {len(entries):,}")
    print(f"💾 Estimated size: ~{estimated_mb:.1f} MB")
    print(f"📁 Download location: {OUTPUT_ROOT}")
    
    # Check if any species has many images
    species_counts = defaultdict(int)
    for entry in entries:
        species_counts[entry["animal_species"]] += 1
    
    max_images = max(species_counts.values())
    if max_images > 1000:
        print(f"⚠️  Some species have {max_images:,} images")
    
    while True:
        try:
            print("\n💡 Options:")
            print("  Y - Start download")
            print("  N - Cancel and select more species") 
            print("  L - Limit images per species")
            print("  Q - Quit program")
            
            confirm = get_user_input("🚀 Your choice (Y/N/L/Q): ").upper()
            
            if confirm == 'Y':
                return download_with_limit(entries, None)
            elif confirm == 'N':
                return False
            elif confirm == 'L':
                while True:
                    try:
                        limit_input = get_user_input("📊 Max images per species (e.g., 2000): ").strip()
                        if not limit_input:
                            print("Please enter a number")
                            continue
                        limit = int(limit_input)
                        if limit <= 0:
                            print("Please enter a positive number")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number")
                
                return download_with_limit(entries, limit)
            elif confirm == 'Q':
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("Please enter Y, N, L, or Q")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user.")
            sys.exit(1)

def normalize_common_name(name):
    """Normalize common name for grouping"""
    # Extract part before parentheses and normalize
    if '(' in name:
        common_part = name.split('(')[0].strip()
    else:
        common_part = name.strip()
    
    # Basic normalization
    normalized = common_part.lower()
    
    # Remove trailing 's' for basic plurals (but not for words ending in 'ss')
    if normalized.endswith('s') and not normalized.endswith('ss') and len(normalized) > 3:
        normalized = normalized[:-1]
    
    return normalized

def group_species_for_limiting(entries):
    """Group species by Latin name or normalized common name"""
    # Group by Latin name first (most reliable)
    latin_groups = defaultdict(list)
    no_latin = []
    
    for entry in entries:
        species_name = entry["animal_species"]
        latin_match = re.search(r'\(([^)]+)\)', species_name)
        
        if latin_match:
            latin_name = latin_match.group(1).strip()
            # Skip generic Latin names like just "aves"
            if latin_name.lower() not in ['aves', 'mammalia', 'reptilia', 'amphibia']:
                latin_groups[latin_name].append(entry)
            else:
                no_latin.append(entry)
        else:
            no_latin.append(entry)
    
    # Group remaining by normalized common name
    common_groups = defaultdict(list)
    for entry in no_latin:
        species_name = entry["animal_species"]
        normalized = normalize_common_name(species_name)
        common_groups[normalized].append(entry)
    
    # Combine groups
    all_groups = {}
    
    # Add Latin name groups
    for latin_name, group_entries in latin_groups.items():
        if len(group_entries) > 1:
            # Multiple entries with same Latin name
            species_names = [e["animal_species"] for e in group_entries]
            group_name = f"Multiple variants of {latin_name}"
            all_groups[group_name] = group_entries
        else:
            # Single entry with Latin name
            all_groups[group_entries[0]["animal_species"]] = group_entries
    
    # Add common name groups  
    for normalized, group_entries in common_groups.items():
        if len(group_entries) > 1:
            # Multiple similar common names
            species_names = [e["animal_species"] for e in group_entries]
            # Use the shortest name as representative
            rep_name = min(species_names, key=len)
            group_name = f"{rep_name} (merged variants)"
            all_groups[group_name] = group_entries
        else:
            # Single entry
            all_groups[group_entries[0]["animal_species"]] = group_entries
    
    return all_groups

def download_with_limit(entries, limit):
    """Download with optional per-species limit and smart grouping"""
    import random
    
    if limit:
        print(f"\n🎲 Applying limit of {limit:,} images per species/group...")
        
        # Group species intelligently
        species_groups = group_species_for_limiting(entries)
        
        # Apply limit with random sampling
        limited_entries = []
        for group_name, group_entries in species_groups.items():
            if len(group_entries) > limit:
                sampled = random.sample(group_entries, limit)
                limited_entries.extend(sampled)
                # Remove the detailed prints - just show count
            else:
                limited_entries.extend(group_entries)
        
        entries = limited_entries
        print(f"🎯 Final count: {len(entries):,} images")
    
    # Rest of download logic stays the same...
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create species directories
    species_dirs = {}
    for entry in entries:
        species_name = entry["animal_species"]
        if species_name not in species_dirs:
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', species_name)
            species_dir = output_root / clean_name
            species_dir.mkdir(parents=True, exist_ok=True)
            species_dirs[species_name] = species_dir
    
    def get_unique_filename(original_filename, uuid, species_dir):
        """Generate unique filename with UUID"""
        original_path = Path(original_filename)
        stem = original_path.stem
        suffix = original_path.suffix
        
        new_filename = f"{stem}_{uuid}{suffix}"
        base_path = species_dir / new_filename
        
        if not base_path.exists():
            return base_path
        
        counter = 1
        while True:
            newer_filename = f"{stem}_{uuid}({counter}){suffix}"
            new_path = species_dir / newer_filename
            if not new_path.exists():
                return new_path
            counter += 1
    
    # Download with progress bar
    session = build_session()
    successes = 0
    already_exists = 0
    failures = 0
    failed_rows = []
    species_metadata = defaultdict(list)
    
    print(f"\n🔄 Starting download of {len(entries):,} images...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        
        for entry in entries:
            url = entry["url_to_original"]
            species_name = entry["animal_species"]
            species_dir = species_dirs[species_name]
            uuid = entry["UUID"]
            
            original_filename = entry.get("filename", f"{uuid}.jpg")
            output_path = get_unique_filename(original_filename, uuid, species_dir)
            
            future = executor.submit(download_with_validation, url, output_path, session)
            futures[future] = (entry, output_path)
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc="🔄 Downloading"):
            entry, output_path = futures[future]
            species_name = entry["animal_species"]
            
            try:
                success, message = future.result()
                if success:
                    if message == "exists-valid":
                        already_exists += 1
                    else:
                        successes += 1
                    
                    metadata_entry = {
                        "uuid": entry["UUID"],
                        "filename": output_path.name,
                        "original_filename": entry.get("filename", ""),
                        "url": entry["url_to_original"],
                        "dataset_name": entry.get("dataset_name", "")
                    }
                    species_metadata[species_name].append(metadata_entry)
                    
                else:
                    failures += 1
                    failed_rows.append(f"{entry['filename']}\t{message}\t{entry['url_to_original']}")
                    
            except Exception as e:
                failures += 1
                failed_rows.append(f"{entry['filename']}\tEXCEPTION {e}\t{entry['url_to_original']}")
    
    # Write metadata files
    print(f"\n📝 Writing metadata files...")
    for species_name, metadata_list in species_metadata.items():
        species_dir = species_dirs[species_name]
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', species_name)
        metadata_file = species_dir / f"{clean_name}.json"
        
        projects = list(set(entry.get("project_name", "Unknown") for entry in entries if entry["animal_species"] == species_name))
        datasets = list(set(entry.get("dataset_name", "Unknown") for entry in entries if entry["animal_species"] == species_name))
        animal_classes = list(set(entry.get("animal_class", "Unknown") for entry in entries if entry["animal_species"] == species_name))
        
        species_info = {
            "species_name": species_name,
            "total_images": len(metadata_list),
            "successful_downloads": len([m for m in metadata_list if not m.get("download_status", "").startswith(("FAILED", "EXCEPTION"))]),
            "failed_downloads": len([m for m in metadata_list if m.get("download_status", "").startswith(("FAILED", "EXCEPTION"))]),
            "projects": projects,
            "datasets": datasets,  
            "animal_classes": animal_classes,
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images": sorted(metadata_list, key=lambda x: x["filename"])
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(species_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Download complete!")
    print(f"📈 Results:")
    print(f"  • Successfully downloaded: {successes:,}")
    print(f"  • Already existed: {already_exists:,}")
    print(f"  • Failed: {failures:,}")
    
    if failures > 0:
        failed_log = output_root / "failed_downloads.txt"
        with open(failed_log, "w", encoding="utf-8") as f:
            for row in failed_rows:
                f.write(row + "\n")
        print(f"⚠️  Failed downloads logged to: {failed_log}")
    
    return True

def main():
    try:
        print("🚀 ANIMAL DATASET INTERACTIVE DOWNLOADER")
        print("="*50)
        
        # Load data
        categories = load_categories()
        catalog = load_catalog()
        
        print(f"\n✅ Loaded taxonomy with {len(categories)} classes")
        print(f"✅ Loaded catalog with {catalog['metadata']['total_images']:,} images")
        
        # Main loop to allow multiple selections
        all_selected_species = []
        total_consolidated_count = 0
        
        while True:
            # Interactive selection
            selection_result = collect_user_selections(categories, catalog)
            
            if not selection_result:
                if not all_selected_species:
                    print("❌ No species selected. Exiting.")
                    return
                break
            
            selected_species, consolidated_count = selection_result
            
            # Add to cumulative selections
            all_selected_species.extend(selected_species)
            total_consolidated_count += consolidated_count
            
            print(f"\n🎯 Added {consolidated_count} species to selection")
            print(f"🎯 Total selected: {total_consolidated_count} species")
            
            # Analyze and show statistics
            filtered_entries = analyze_selections(all_selected_species, catalog)
            
            # Try to download
            download_result = download_images(filtered_entries)
            
            if download_result is False:
                # User chose to continue selecting more species
                continue
            elif download_result is True:
                # Download completed successfully
                break
        
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        print("Make sure you've run the catalog generator first!")
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()