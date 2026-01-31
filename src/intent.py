"""Intent parser for natural language queries."""

import re
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Types of user intents."""

    DESCRIBE = "describe"  # "What is in front of me?"
    SEARCH = "search"  # "Where is my phone?"
    DETECT = "detect"  # "What objects are here?"
    COUNT = "count"  # "How many chairs?"
    ALERT = "alert"  # Passive mode (no query)
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Parsed intent from user query."""

    type: IntentType
    target: str | None = None  # Target object for search/count
    raw_query: str = ""


class IntentParser:
    """Parse natural language queries into structured intents."""

    # Patterns for intent classification
    DESCRIBE_PATTERNS = [
        r"what('?s| is)? (in front|ahead|around|here|there)",
        r"describe (the )?(scene|room|area|surroundings?|view)",
        r"what (do|can) (you|i) see",
        r"tell me (about|what)",
        r"what is (this|that)",
        r"what('?s| is) (happening|going on)",
        r"what are you (seeing|looking at)",
        r"explain (the )?(scene|view|image)",
        r"show me",
        r"help me (see|understand)",
        r"i('?m| am) blind",
        r"guide me",
        r"navigate",
    ]

    SEARCH_PATTERNS = [
        r"where('?s| is)? (my |the |a )?(?P<target>\w+)",
        r"find (my |the |a )?(?P<target>\w+)",
        r"locate (my |the |a )?(?P<target>\w+)",
        r"look for (my |the |a )?(?P<target>\w+)",
        r"can you (find|see|spot) (my |the |a )?(?P<target>\w+)",
        r"is there (a |any )?(?P<target>\w+)",
        r"do you see (a |any |my )?(?P<target>\w+)",
        r"search for (my |the |a )?(?P<target>\w+)",
    ]

    DETECT_PATTERNS = [
        r"what objects",
        r"what things",
        r"what items",
        r"list (all )?(objects|things|items)",
        r"detect (all )?(objects|everything)",
        r"identify (all )?(objects|things)",
        r"what('?s| is) (all )?around",
        r"scan (the )?(room|area|scene)",
    ]

    COUNT_PATTERNS = [
        r"how many (?P<target>\w+)",
        r"count (the |all )?(?P<target>\w+)",
        r"number of (?P<target>\w+)",
        r"are there (any |multiple )?(?P<target>\w+)",
    ]

    # Keywords that suggest describing (fallback)
    DESCRIBE_KEYWORDS = [
        "what", "see", "look", "view", "scene", "front", "ahead",
        "around", "help", "tell", "describe", "explain", "show",
        "saying", "happening", "going", "there"
    ]

    def __init__(self) -> None:
        # Compile patterns for efficiency
        self._describe_re = [re.compile(p, re.IGNORECASE) for p in self.DESCRIBE_PATTERNS]
        self._search_re = [re.compile(p, re.IGNORECASE) for p in self.SEARCH_PATTERNS]
        self._detect_re = [re.compile(p, re.IGNORECASE) for p in self.DETECT_PATTERNS]
        self._count_re = [re.compile(p, re.IGNORECASE) for p in self.COUNT_PATTERNS]

    def parse(self, query: str) -> Intent:
        """Parse a natural language query into an Intent.

        Args:
            query: User's natural language query.

        Returns:
            Intent object with type and optional target.
        """
        query = query.strip()
        if not query:
            return Intent(type=IntentType.ALERT, raw_query=query)

        # Check describe patterns
        for pattern in self._describe_re:
            if pattern.search(query):
                return Intent(type=IntentType.DESCRIBE, raw_query=query)

        # Check count patterns (before search to catch "how many X")
        for pattern in self._count_re:
            match = pattern.search(query)
            if match:
                target = match.group("target") if "target" in match.groupdict() else None
                return Intent(type=IntentType.COUNT, target=target, raw_query=query)

        # Check search patterns
        for pattern in self._search_re:
            match = pattern.search(query)
            if match:
                target = match.group("target") if "target" in match.groupdict() else None
                return Intent(type=IntentType.SEARCH, target=target, raw_query=query)

        # Check detect patterns
        for pattern in self._detect_re:
            if pattern.search(query):
                return Intent(type=IntentType.DETECT, raw_query=query)

        # Fallback: check for describe keywords
        query_lower = query.lower()
        for keyword in self.DESCRIBE_KEYWORDS:
            if keyword in query_lower:
                return Intent(type=IntentType.DESCRIBE, raw_query=query)

        # If query is a question (starts with question word), default to describe
        question_words = ["what", "how", "can", "could", "will", "is", "are", "do", "does"]
        first_word = query_lower.split()[0] if query_lower.split() else ""
        if first_word in question_words or query.endswith("?"):
            return Intent(type=IntentType.DESCRIBE, raw_query=query)

        # Default to unknown
        return Intent(type=IntentType.UNKNOWN, raw_query=query)

    def extract_object(self, query: str) -> str | None:
        """Extract object name from a query.

        Args:
            query: User's query string.

        Returns:
            Object name or None if not found.
        """
        # Common patterns for object extraction
        patterns = [
            r"(?:where is|find|locate|look for) (?:my |the |a )?(\w+)",
            r"(?:how many|count) (\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return None


def parse_intent(query: str) -> Intent:
    """Convenience function to parse a query.

    Args:
        query: User's natural language query.

    Returns:
        Intent object.
    """
    parser = IntentParser()
    return parser.parse(query)


if __name__ == "__main__":
    parser = IntentParser()

    test_queries = [
        # Describe
        "What is in front of me?",
        "What's ahead?",
        "Describe the scene",
        "What do you see?",
        "Tell me about my surroundings",
        # Search
        "Where is my phone?",
        "Find the door",
        "Where's the chair?",
        "Can you find my keys?",
        "Locate the table",
        # Detect
        "What objects are here?",
        "List all objects",
        "Detect everything",
        # Count
        "How many chairs?",
        "Count the people",
        "Number of doors",
        # Unknown
        "Hello",
        "Help me",
        "",
    ]

    print("Intent Parser Test Results:")
    print("-" * 60)
    for query in test_queries:
        intent = parser.parse(query)
        print(f"Query: '{query}'")
        print(f"  → Type: {intent.type.value}, Target: {intent.target}")
        print()
