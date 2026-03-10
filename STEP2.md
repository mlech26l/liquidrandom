# Step 2 of liquidrandom

In the second phase of liquidrandom, we want to add an additional seed data type in the form of MCPs, tools, and functions for agentic workflows.

## Format

Use the OpenAI compatible format for tool and functions as a data model.

## Additional randomization needed:

The same function can be implemented with differently named tools or tools/function that have different parameter (names, formats, types etc).
Consider the following example:

```python
def search_flights(origin, destination, date):
    # code to search for flights
    pass

def find_flights(origin_city, destination_city, departure_date, return_date=None):
    # code to search for flights
    pass

def flight_lookup(class_type, from_code, to_code, year, month, day):
    # code to search for flights
    pass
....
```

The above three python functions illustrate the same functionality can be realized with different parameter names, formats, and types. This is a common occurrence in real-world codebases and can lead to confusion when trying to understand or use the functions.
Therefore, we don't to create a single function for each tool we can to create M functions for each tool that differ in a similar fashion as the explanation and example above. This will allow us to test the robustness of our agentic workflows and ensure that they can handle a variety of different inputs and formats.

M=8

In order to achieve this, the LLM that generates the functions needs the previously generated functions as input to create variations of them. This way we can ensure that the variations are realistic and reflect the diversity of real-world codebases.

## Groups of tools/functions:

Tools and functions do not come along but as a group. In the example above for searching flight, we also most likely have a tool for booking flights, a tool for checking flight status, and a tool for canceling flights. These tools are all related and are often used together in an agentic workflow. Therefore, we want to create groups of tools/functions that are related and can be used together in an agentic workflow. 

It is important because there dependency between these tools/functions, eg. the return type of the first tool is an input for the second tool etc.

It probably makes sense to generate the group first and then the variations, except if you strongly believe otherwise, eg. if you think that generating a single tool and then iteratively related tools is better then we should go with this instead.

## Types of tools

We want reuse the taxonomy tree from the first phase of liquidrandom, but with agentic workflows in mind. Use the websearch tool to look for categories of MCP servers and LLM tools to create the initial root nodes for the taxonomy tree. 
We probably do not need too much depth in the tree though, eg. 2 or 3 levels.

## Checking

Similar to phase 1 we want to also feed the output into the LLM again to check for issues.

## Conclusion

- Tools, MCPs, and function as additional seed data types for liquidrandom
- We want to group related tools/functions together to create agentic workflows
- For each tool/function we want to create M=8 variations that differ in parameter names, formats, and types
- Seed the tree by doing a websearch
- Use the same LLM to check output for issues
