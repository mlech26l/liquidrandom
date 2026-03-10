import json

from liquidrandom.models import ToolFunction, ToolGroup, ToolVariation


def test_tool_variation_from_dict_and_str() -> None:
    data = {
        "name": "search_flights",
        "description": "Search for available flights",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["origin", "destination", "date"],
        },
        "returns": {
            "type": "object",
            "properties": {
                "flights": {"type": "array", "items": {"type": "object"}},
            },
        },
    }
    v = ToolVariation.from_dict(data)
    assert v.name == "search_flights"
    assert v.parameters["type"] == "object"
    assert "origin" in v.parameters["properties"]
    assert v.returns["type"] == "object"
    s = str(v)
    assert "search_flights" in s
    assert "Search for available flights" in s


def test_tool_variation_from_dict_handles_none() -> None:
    data = {
        "name": "test",
        "description": "test",
        "parameters": None,
        "returns": None,
    }
    v = ToolVariation.from_dict(data)
    assert v.parameters == {}
    assert v.returns == {}


def test_tool_function_from_dict_and_str() -> None:
    data = {
        "canonical_name": "search_flights",
        "description": "Search for available flights",
        "variations": [
            {
                "name": "search_flights",
                "description": "Search for available flights",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "returns": {"type": "object", "properties": {}},
            },
            {
                "name": "find_flights",
                "description": "Find flights matching criteria",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "returns": {"type": "object", "properties": {}},
            },
        ],
    }
    f = ToolFunction.from_dict(data)
    assert f.canonical_name == "search_flights"
    assert len(f.variations) == 2
    assert f.variations[1].name == "find_flights"
    s = str(f)
    assert "search_flights" in s
    assert "Variations (2)" in s


def test_tool_function_from_dict_handles_none_variations() -> None:
    data = {
        "canonical_name": "test",
        "description": "test",
        "variations": None,
    }
    f = ToolFunction.from_dict(data)
    assert f.variations == []


def test_tool_group_from_dict_and_str() -> None:
    data = {
        "domain": "Flight Booking",
        "description": "Tools for searching and booking flights",
        "taxonomy_path": "Travel > Flights > Booking",
        "tools": [
            {
                "canonical_name": "search_flights",
                "description": "Search for flights",
                "variations": [
                    {
                        "name": "search_flights",
                        "description": "Search for flights",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "returns": {"type": "object", "properties": {}},
                    }
                ],
            },
            {
                "canonical_name": "book_flight",
                "description": "Book a specific flight",
                "variations": [
                    {
                        "name": "book_flight",
                        "description": "Book a specific flight",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "returns": {"type": "object", "properties": {}},
                    }
                ],
            },
        ],
    }
    g = ToolGroup.from_dict(data)
    assert g.domain == "Flight Booking"
    assert g.taxonomy_path == "Travel > Flights > Booking"
    assert len(g.tools) == 2
    assert g.tools[0].canonical_name == "search_flights"
    s = str(g)
    assert "Flight Booking" in s
    assert "search_flights" in s
    assert "book_flight" in s
    assert "Tools (2)" in s


def test_tool_group_from_dict_with_tools_json_string() -> None:
    tools_data = [
        {
            "canonical_name": "get_user",
            "description": "Get user info",
            "variations": [
                {
                    "name": "get_user",
                    "description": "Get user info",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "returns": {"type": "object", "properties": {}},
                }
            ],
        }
    ]
    data = {
        "domain": "User Management",
        "description": "Manage users",
        "taxonomy_path": "Identity > Users",
        "tools_json": json.dumps(tools_data),
    }
    g = ToolGroup.from_dict(data)
    assert g.domain == "User Management"
    assert len(g.tools) == 1
    assert g.tools[0].canonical_name == "get_user"


def test_tool_group_from_dict_handles_none_tools() -> None:
    data = {
        "domain": "Empty",
        "description": "Empty group",
        "taxonomy_path": "Test",
        "tools": None,
    }
    g = ToolGroup.from_dict(data)
    assert g.tools == []
