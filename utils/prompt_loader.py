from pathlib import Path


def load_prompt(path: str) -> str:
    '''
    Load a prompt template from a text file.

    Parameters
    ----------
    path : str
        Path to the prompt file.

    Returns
    -------
    str
        Prompt template as a string.
    '''
    return Path(path).read_text(encoding='utf-8')
