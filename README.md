## **Prerequisites**

1. **Python Installation** :

* Ensure Python 3.8 or higher is installed on your system.
* Install required libraries using the following commands:

  **pip** **install** **opencv-python** **opencv-python-headless** **numpy** **matplotlib** **torch** **torchvision** **scikit-learn** **pillow**

1. **Project Structure** :

* Ensure the project directory contains the following:

  **c:\PythonProjects\CVProj1\**

  **├── [main_reader.py](**http://_vscodecontentref_/1**)**

  **├── [board.py](**http://_vscodecontentref_/2**)**

  **├── [pipelineFunctions.py](**http://_vscodecontentref_/3**)**

  **└── moves_output/  # This will be created **automatically

---

## **How to Run**

1. **Navigate to the Project Directory** : Open a terminal or command prompt and navigate to the project folder:
2. **Run the Script** : Execute the `main_reader.py` file with the folder containing images as an argument:

   **python** ****main_reader.py `path/to/your/image/folder`****

   Replace `path/to/your/image/folder` with the path to the folder containing the images to be processed.

1. **Expected Output** :

* The script will process images from the specified folder.
* For each move, it will:
  * Detect and classify tiles.
  * Calculate the score.
  * Save the results in the `moves_output/` directory.

---

## **Output Files**

1. **Location** :

* The output files are saved in the `moves_output/` directory.

1. **File Format** :

* Each file is named as `1_XX.txt`, where `XX` is the move number (e.g., `1_00.txt`, `1_01.txt`).

1. **File Content** :

* Each file contains:
  * Tile positions and their classifications (e.g., `10D 4G`).
  * The score for the move as the last line.

    **Example** :

   **10D 4G**

   **11D 5G**

   **13D 2G**

   **10**
