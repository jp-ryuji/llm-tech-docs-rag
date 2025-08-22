from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.readers.file import MarkdownReader


# pylint: disable=too-few-public-methods
class DocumentLoader:
    """Loads documents from the specified data directory."""

    def __init__(self, data_path: str):
        """
        Initializes the DocumentLoader.

        Args:
            data_path: The path to the directory containing data files.
        """
        self.data_path = Path(data_path)

    def load_documents(self) -> List[Document]:
        """
        Loads all markdown documents from the data directory and its subdirectories.

        Returns:
            A list of loaded Document objects.
        """
        documents = []

        md_reader = MarkdownReader()
        for md_file in self.data_path.rglob("*.md"):
            try:
                docs = md_reader.load_data(file=str(md_file))
                for doc in docs:
                    doc.metadata.update(
                        {
                            "source": str(md_file),
                            "file_type": "markdown",
                            "section": self._extract_section(md_file),
                        }
                    )
                documents.extend(docs)
            except IOError as e:
                print(f"Error loading {md_file}: {e}")

        return documents

    def _extract_section(self, file_path: Path) -> str:
        """
        Extracts a section name from the file path.

        The section is assumed to be the name of the parent directory.

        Args:
            file_path: The path to the file.

        Returns:
            The extracted section name, or "general" if not found.
        """
        parts = file_path.parts
        if len(parts) > 2:
            return parts[-2]  # parent directory name
        return "general"
