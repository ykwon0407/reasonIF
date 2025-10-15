from . import instruction_checker

_LANGUAGE = "language:"
_LENGTH = "length_constraint_checkers:"
_CHANGE_CASES = "change_case:"
_STARTEND = "startend:"
_FORMAT = "detectable_format:"
_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    # realistic
    _LANGUAGE + "reasoning_language": instruction_checker.ReasoningLanguageChecker,
    _LENGTH + "number_words": instruction_checker.NumberOfWords,
    _CHANGE_CASES + "english_capital": instruction_checker.CapitalLettersEnglishChecker,
    _STARTEND + "end_checker": instruction_checker.EndChecker,

    # artificial 
    _FORMAT + "json_format": instruction_checker.JsonFormat,
    _PUNCTUATION + "no_comma": instruction_checker.CommaChecker,
}

