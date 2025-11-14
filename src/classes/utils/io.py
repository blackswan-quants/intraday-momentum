import pickle
import re
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Ascends the directory hierarchy from the current file's location until it 
    finds the 'src' folder, and then returns its parent directory (the project root).

    Raises:
        FileNotFoundError: If the 'src' directory cannot be found within the hierarchy.

    Returns:
        Path: The absolute path to the project root directory (e.g., 'intraday-momentum/').
    """
    # Start checking from the location of the current file (io.py)
    current_path = Path(__file__).resolve()
    
    # Traverse up the hierarchy until 'src' is found or the root is reached
    while current_path.name != 'src' and current_path != current_path.parent:
        current_path = current_path.parent
        
    if current_path.name == 'src':
        # If 'src' is found, the parent is the project root (e.g., .../intraday-momentum)
        return current_path.parent
    else:
        # Fallback if the structure is unexpected
        raise FileNotFoundError("Could not find the project root (directory containing 'src').")


class PickleHelper:
    """
    Utility class for serializing and deserializing Python objects using the pickle module.
    """
    
    def __init__(self, obj: Any) -> None:
        """
        Initializes the PickleHelper with the object to be managed.

        Args:
            obj (Any): The object (e.g., a pandas DataFrame) to be pickled or managed.
        """
        self.obj = obj

    def pickle_dump(self, filename: str) -> None:
        """
        Serializes the managed object and saves it to the 'data/cleaned/' directory 
        within the project root using the pickle format.

        Args:
            filename (str): The name of the file (e.g., "cleaned_data"). The ".pkl" 
                            extension is automatically added if missing.
        """
        
        # Ensure the filename ends with .pkl
        if not re.search(r"^.*\.pkl$", filename):
            filename += ".pkl"

        # Dynamically determine the project root
        project_root: Path = get_project_root()
        
        # Construct the full absolute path: [ROOT]/data/cleaned/[filename.pkl]
        file_path: Path = project_root / "data" / "cleaned" / filename
        
        try:
            # Create the necessary directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as f:
                pickle.dump(self.obj, f)
            logger.info(f"Object successfully pickled and saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving object to {file_path}: {e}")
            raise 
            

    @staticmethod
    def pickle_load(filename: str) -> Optional['PickleHelper']:
        """
        Loads a serialized object from the 'data/cleaned/' directory within the project root.

        Args:
            filename (str): The name of the file from which the object will be loaded. 
                            The ".pkl" extension is added if missing.

        Returns:
            Optional[PickleHelper]: A PickleHelper object with the loaded data, 
                                    or None if the file is not found or an error occurs.
        """
        
        # Ensure the filename ends with .pkl
        if not re.search(r"^.*\.pkl$", filename):
            filename += ".pkl"

        try:
            # Dynamically determine the project root
            project_root: Path = get_project_root()
        except FileNotFoundError:
            logger.error("Project root not found, cannot determine load path.")
            return None

        # Construct the full absolute path for loading
        file_path: Path = project_root / "data" / "cleaned" / filename

        try:
            with open(file_path, "rb") as f:
                
                loaded_obj: Any = pickle.load(f)
                logger.info(f"Object successfully loaded from: {file_path}")
                # Return the loaded object wrapped in a new PickleHelper instance
                return PickleHelper(loaded_obj)
                
        except FileNotFoundError:
            
            logger.error(f"The file '{file_path}' does not exist.")
            return None
            
        except Exception as e:
            logger.error(f"An error occurred while loading object from {file_path}: {e}")
            return None