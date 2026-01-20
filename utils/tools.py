hyponym_only = [
    {
        "type": "function",
        "function": {
            "name": "get_hyponyms",
            "description": "Navigate the RuWordNet taxonomy by retrieving hyponyms (more specific concepts) of a given synset. Returns formatted markdown with: synset name, ID, associated words, and list of child hyponyms. When node_id is null, returns all root nodes (top-level concepts). Use this to explore the taxonomy tree level by level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": ["string", "null"],
                        "description": "The synset ID to get hyponyms for. Use 'null' or null to retrieve all root nodes (top-level concepts in the taxonomy). Use specific synset ID like '123456-N' to get its children.",
                    },
                },
                "required": ["node_id"],
            },
        },
    }
]
hypernym_only = [
    {
        "type": "function",
        "function": {
            "name": "get_hypernyms",
            "description": "Navigate the RuWordNet taxonomy by retrieving hypernyms (more abstract concepts) of a given synset. Returns formatted markdown with: synset name, ID, associated words, and list of parent hypernyms. Use this to explore the taxonomy tree level by level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": ["string"],
                        "description": "The synset ID to get hypernyms for. Use specific synset ID like '123456-N' to get its parents.",
                    },
                },
                "required": ["node_id"],
            },
        },
    }
]
tools = {
    'hyponym_only': hyponym_only,
    'hypernym_only': hypernym_only
}

__all__ = ['tools', 'hyponym_only', 'hypernym_only']