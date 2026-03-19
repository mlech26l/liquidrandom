"""Category configurations for seed data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    display_name: str
    fields: list[FieldSpec]
    taxonomy_seed_prompt: str
    generation_prompt_template: str
    specificity_guidance: str


CATEGORY_CONFIGS: dict[str, CategoryConfig] = {
    "persona": CategoryConfig(
        name="persona",
        display_name="Personas",
        fields=[
            FieldSpec("name", "string", "Full name of the persona"),
            FieldSpec("age", "integer", "Age in years"),
            FieldSpec("gender", "string", "Gender identity"),
            FieldSpec("occupation", "string", "Current occupation"),
            FieldSpec("nationality", "string", "Nationality or cultural background"),
            FieldSpec("personality_traits", "array of strings", "3-5 distinct personality traits"),
            FieldSpec("background", "string", "Brief background story (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of persona archetypes organized by life stage, "
            "cultural background, and occupation cluster. Each leaf should be a "
            "highly specific persona archetype (e.g. 'retired Japanese calligraphy "
            "teacher living in rural Hokkaido' not just 'elderly person')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific personas that fit the category: {leaf_path}\n\n"
            "Each persona must be a complete, believable individual with specific details. "
            "Avoid generic descriptions. Each persona should feel like a real person with "
            "a unique story.\n\n"
            "Previously generated personas for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): Full realistic name appropriate to their background\n"
            "- age (integer): Specific age\n"
            "- gender (string): Gender identity\n"
            "- occupation (string): Specific occupation\n"
            "- nationality (string): Nationality or cultural background\n"
            "- personality_traits (array of strings): 3-5 distinct personality traits\n"
            "- background (string): Brief but specific background story (2-3 sentences)"
        ),
        specificity_guidance="Each persona should be a specific, believable individual, not a generic archetype.",
    ),
    "job": CategoryConfig(
        name="job",
        display_name="Jobs",
        fields=[
            FieldSpec("job_category", "string", "Broad job category (e.g. 'Data Science', 'DevOps', 'Clinical Research')"),
            FieldSpec("sector", "string", "Broad sector (e.g. 'Financial Services', 'Healthcare', 'Government')"),
            FieldSpec("title", "string", "Specific job title"),
            FieldSpec("industry", "string", "Specific industry niche"),
            FieldSpec("description", "string", "What the job entails (2-3 sentences)"),
            FieldSpec("required_skills", "array of strings", "3-5 specific required skills"),
            FieldSpec("experience_level", "string", "Experience level (entry/mid/senior/lead/executive)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of professions organized by industry sector, "
            "specialization, and seniority. Each leaf should be a very specific "
            "role (e.g. 'pediatric oncology clinical trial coordinator' not just 'doctor')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific job descriptions that fit the category: {leaf_path}\n\n"
            "Each job must be a real, specific role with concrete details. "
            "Avoid generic job descriptions.\n\n"
            "Previously generated jobs for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- job_category (string): Broad job category (e.g. 'Data Science', 'DevOps', 'Clinical Research') - "
            "this should be broader than the title, applicable to many similar roles\n"
            "- sector (string): Broad economic sector (e.g. 'Financial Services', 'Healthcare', 'Government') - "
            "broader than the specific industry\n"
            "- title (string): Specific job title\n"
            "- industry (string): Specific industry niche\n"
            "- description (string): What the job entails (2-3 sentences)\n"
            "- required_skills (array of strings): 3-5 specific required skills\n"
            "- experience_level (string): one of entry/mid/senior/lead/executive"
        ),
        specificity_guidance="Each job should be a specific role with concrete responsibilities, not a generic title.",
    ),
    "coding_task": CategoryConfig(
        name="coding_task",
        display_name="Coding Tasks",
        fields=[
            FieldSpec("title", "string", "Short title of the task"),
            FieldSpec("language", "string", "Programming language"),
            FieldSpec("difficulty", "string", "easy/medium/hard/expert"),
            FieldSpec("description", "string", "Detailed task description (2-4 sentences). Must be specific enough to understand and execute without ambiguity."),
            FieldSpec("constraints", "array of strings", "2-4 specific constraints. Each constraint must be concrete and measurable."),
            FieldSpec("expected_behavior", "string", "What the solution should do. Be detailed about inputs, outputs, and edge case handling."),
            FieldSpec("follow_up_task", "string", "A feature or extension that can be added as a follow-up. Must be related to the original task and detailed enough to implement. Should NOT be about change requests or edge cases."),
            FieldSpec("change_request", "string", "A change request to the original task, useful for testing ability to modify existing solutions. Be specific about what should change and why."),
            FieldSpec("edge_cases", "array of strings", "2-4 edge cases or additional input types that make the task more challenging. If the task doesn't have natural edge cases, describe unusual inputs to support."),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of programming tasks organized by domain "
            "(algorithms, systems, web, data, etc.), sub-domain, and specific "
            "technique. Each leaf should be a very specific task type "
            "(e.g. 'implement B+ tree with bulk loading optimization' not just "
            "'data structure implementation')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific coding tasks that fit the category: {leaf_path}\n\n"
            "Each task must be a concrete programming challenge with clear, detailed requirements. "
            "The description must be specific enough that a developer can understand and implement it "
            "without asking clarifying questions. Constraints and expected behavior must be concrete and "
            "measurable. Avoid vague or overly broad tasks.\n\n"
            "Previously generated tasks for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- title (string): Short, descriptive title\n"
            "- language (string): Programming language\n"
            "- difficulty (string): one of easy/medium/hard/expert\n"
            "- description (string): Detailed task description (2-4 sentences), specific enough to implement\n"
            "- constraints (array of strings): 2-4 specific, measurable constraints\n"
            "- expected_behavior (string): Detailed description of what the solution should do\n"
            "- follow_up_task (string): A feature or extension that could be added later. Must be a genuine "
            "feature extension related to the original task, NOT a change request or edge case handling.\n"
            "- change_request (string): A specific change request to modify the original requirements. "
            "Describe what should change and why.\n"
            "- edge_cases (array of strings): 2-4 edge cases or unusual inputs that make the task more challenging"
        ),
        specificity_guidance="Each task should specify exact algorithms, data structures, or techniques needed.",
    ),
    "math_category": CategoryConfig(
        name="math_category",
        display_name="Math Categories",
        fields=[
            FieldSpec("broad_topic", "string", "Broad math topic (e.g. 'Equation Solving', 'Optimization', 'Number Theory')"),
            FieldSpec("name", "string", "Specific math topic name"),
            FieldSpec("field", "string", "Mathematical field"),
            FieldSpec("description", "string", "What this topic covers (2-3 sentences)"),
            FieldSpec("example_problems", "array of strings", "2-3 concrete example problems"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of mathematical topics organized by field "
            "(algebra, analysis, geometry, combinatorics, etc.), sub-field, "
            "and specific technique or theorem family. Each leaf should be "
            "a very specific topic (e.g. 'solving recurrence relations using "
            "generating functions' not just 'combinatorics')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific math topics that fit the category: {leaf_path}\n\n"
            "Each topic must be a concrete mathematical concept with specific examples. "
            "Avoid broad categories.\n\n"
            "Previously generated topics for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- broad_topic (string): Broad math topic category (e.g. 'Equation Solving', 'Optimization', "
            "'Proof Techniques') - should be applicable to many related specific topics\n"
            "- name (string): Specific math topic name\n"
            "- field (string): Mathematical field\n"
            "- description (string): What this topic covers (2-3 sentences)\n"
            "- example_problems (array of strings): 2-3 concrete example problems"
        ),
        specificity_guidance="Each topic should name specific techniques, theorems, or problem types.",
    ),
    "writing_style": CategoryConfig(
        name="writing_style",
        display_name="Writing Styles",
        fields=[
            FieldSpec("category", "string", "Broad writing style category (e.g. 'Academic Writing', 'Journalistic', 'Creative Fiction', 'Technical')"),
            FieldSpec("name", "string", "Distinctive name for the style"),
            FieldSpec("tone", "string", "Overall tone"),
            FieldSpec("characteristics", "array of strings", "3-5 stylistic characteristics"),
            FieldSpec("description", "string", "Description of the style (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of writing styles organized by register "
            "(formal/informal), purpose (persuasive, informative, creative, etc.), "
            "and medium (essay, social media, technical documentation, etc.). "
            "Each leaf should be a specific but broadly applicable style "
            "(e.g. 'conversational tech blog post' not 'Victorian epistolary prose "
            "with satirical undertones about industrial economics')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, specific writing styles that fit the category: {leaf_path}\n\n"
            "Each style must be distinctive and well-characterized. Styles should be broadly "
            "applicable across domains, not tied to a specific niche or historical period. "
            "Avoid hyper-specific styles that only apply to one domain.\n\n"
            "Previously generated styles for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- category (string): Broad writing style category (e.g. 'Academic Writing', 'Journalistic', "
            "'Creative Fiction', 'Technical') - applicable to many similar styles\n"
            "- name (string): Distinctive name for the style\n"
            "- tone (string): Overall tone\n"
            "- characteristics (array of strings): 3-5 stylistic characteristics\n"
            "- description (string): Description of the style (2-3 sentences)"
        ),
        specificity_guidance="Each style should be distinctive enough that two different writers would produce noticeably different text, but broad enough to apply across domains.",
    ),
    "scenario": CategoryConfig(
        name="scenario",
        display_name="Scenarios",
        fields=[
            FieldSpec("broad_title", "string", "Broad scenario category (e.g. 'Loan forgiveness scenario', 'Team conflict resolution', 'Emergency response')"),
            FieldSpec("theme", "string", "Thematic category (e.g. 'Financial hardship', 'Workplace dynamics', 'Natural disaster')"),
            FieldSpec("title", "string", "Specific scenario title"),
            FieldSpec("context", "string", "Background context"),
            FieldSpec("setting", "string", "Physical or temporal setting"),
            FieldSpec("stakes", "string", "What is at stake"),
            FieldSpec("description", "string", "Full scenario description (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of real-world scenarios organized by domain "
            "(workplace, personal, crisis, negotiation, etc.), complexity level, "
            "and emotional valence. Each leaf should be a very specific situation "
            "(e.g. 'mediating a dispute between two co-founders over equity split "
            "during Series A' not just 'business conflict')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific scenarios that fit the category: {leaf_path}\n\n"
            "Each scenario must be a vivid, concrete situation with clear stakes. "
            "Avoid generic or abstract scenarios.\n\n"
            "Previously generated scenarios for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- broad_title (string): Broad scenario category (e.g. 'Loan forgiveness scenario', "
            "'Team conflict resolution') - should be applicable if you only need the high-level gist\n"
            "- theme (string): Thematic category (e.g. 'Financial hardship', 'Workplace dynamics')\n"
            "- title (string): Specific, detailed scenario title\n"
            "- context (string): Background context\n"
            "- setting (string): Physical or temporal setting\n"
            "- stakes (string): What is at stake\n"
            "- description (string): Full scenario description (2-3 sentences)"
        ),
        specificity_guidance="Each scenario should paint a vivid picture with specific details about who, what, where, and why.",
    ),
    "domain": CategoryConfig(
        name="domain",
        display_name="Domains",
        fields=[
            FieldSpec("broad_category", "string", "Top-level discipline (e.g. 'Sciences', 'Engineering', 'Humanities', 'Business')"),
            FieldSpec("area", "string", "Broad area within the discipline (e.g. 'Machine Learning', 'Constitutional Law', 'Supply Chain')"),
            FieldSpec("name", "string", "Specific domain name"),
            FieldSpec("parent_field", "string", "Parent field or discipline"),
            FieldSpec("description", "string", "What this domain covers (2-3 sentences)"),
            FieldSpec("key_concepts", "array of strings", "3-5 key concepts"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of knowledge domains organized by discipline "
            "(sciences, humanities, engineering, business, arts, etc.), sub-discipline, "
            "and specialization. Each leaf should be a very specific domain "
            "(e.g. 'pharmacokinetic modeling of biologics in pediatric populations' "
            "not just 'medicine')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific knowledge domains that fit the category: {leaf_path}\n\n"
            "Each domain must be a concrete area of expertise with specific concepts. "
            "Avoid broad or overlapping domains.\n\n"
            "Previously generated domains for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- broad_category (string): Top-level discipline (e.g. 'Sciences', 'Engineering', 'Humanities', "
            "'Business') - the broadest classification\n"
            "- area (string): Broad area within the discipline (e.g. 'Machine Learning', 'Constitutional Law') "
            "- more specific than broad_category but still widely applicable\n"
            "- name (string): Specific domain name\n"
            "- parent_field (string): Parent field or discipline\n"
            "- description (string): What this domain covers (2-3 sentences)\n"
            "- key_concepts (array of strings): 3-5 key concepts specific to this domain"
        ),
        specificity_guidance="Each domain should be narrow enough that an expert would specialize in it.",
    ),
    "science_topic": CategoryConfig(
        name="science_topic",
        display_name="Science Topics",
        fields=[
            FieldSpec("broad_topic", "string", "Broad science topic (e.g. 'Particle Physics', 'Genetics', 'Climate Science')"),
            FieldSpec("name", "string", "Specific topic name"),
            FieldSpec("scientific_field", "string", "Scientific field"),
            FieldSpec("subfield", "string", "Specific subfield"),
            FieldSpec("description", "string", "What this topic covers (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of scientific topics organized by field "
            "(physics, chemistry, biology, earth sciences, etc.), subfield, "
            "and specific phenomenon or research area. Each leaf should be a "
            "very specific topic (e.g. 'regulation of circadian rhythm by "
            "cryptochrome proteins in Drosophila' not just 'biology')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific science topics that fit the category: {leaf_path}\n\n"
            "Each topic must describe a specific scientific phenomenon or research area. "
            "Avoid broad categories.\n\n"
            "Previously generated topics for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- broad_topic (string): Broad science topic (e.g. 'Particle Physics', 'Genetics', "
            "'Climate Science') - applicable to many related specific topics\n"
            "- name (string): Specific topic name\n"
            "- scientific_field (string): Scientific field\n"
            "- subfield (string): Specific subfield\n"
            "- description (string): What this topic covers (2-3 sentences)"
        ),
        specificity_guidance="Each topic should name specific phenomena, mechanisms, or experimental approaches.",
    ),
    "language": CategoryConfig(
        name="language",
        display_name="Languages",
        fields=[
            FieldSpec("category", "string", "Broad language category (e.g. 'Business jargon', 'Youth slang', 'Academic discourse', 'Immigrant dialect')"),
            FieldSpec("name", "string", "Language/variety name"),
            FieldSpec("region", "string", "Geographic region"),
            FieldSpec("register", "string", "Register (formal/informal/colloquial/technical/literary)"),
            FieldSpec("script", "string", "Writing script used"),
            FieldSpec("cultural_notes", "string", "Cultural context and usage notes (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of language varieties and communication styles organized "
            "by broad practical categories. Focus on broadly applicable categories like:\n"
            "- Business jargon, Youth language, Academic discourse\n"
            "- Immigrant dialects, Regional vernaculars, Professional terminology\n"
            "- Digital communication styles, Literary registers\n\n"
            "Each leaf should be a practical, broadly applicable language category "
            "(e.g. 'Business jargon' or 'Chinese immigrant dialect' not "
            "'Cerreto Laghi Resort-Dialect' or 'Litvish Yeshiva Key Hang Hebrew')."
        ),
        generation_prompt_template=(
            "Generate {k} unique language varieties that fit the category: {leaf_path}\n\n"
            "Each entry must describe a practical, broadly applicable language variety. "
            "Focus on categories useful for diverse text generation, not obscure dialects.\n\n"
            "Previously generated entries for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- category (string): Broad language category (e.g. 'Business jargon', 'Youth slang', "
            "'Academic discourse') - broadly applicable category\n"
            "- name (string): Language/variety name\n"
            "- region (string): Geographic region\n"
            "- register (string): Register (formal/informal/colloquial/technical/literary)\n"
            "- script (string): Writing script used\n"
            "- cultural_notes (string): Cultural context and usage notes (2-3 sentences)"
        ),
        specificity_guidance="Each entry should describe a practical language variety useful for text generation, not an obscure or hyper-specific dialect.",
    ),
    "reasoning_pattern": CategoryConfig(
        name="reasoning_pattern",
        display_name="Reasoning Patterns",
        fields=[
            FieldSpec("name", "string", "Pattern name"),
            FieldSpec("category", "string", "Category of reasoning (deductive/inductive/abductive/analogical/etc.)"),
            FieldSpec("description", "string", "How this pattern works (2-3 sentences)"),
            FieldSpec("when_to_use", "string", "When this pattern is most effective (1-2 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of general reasoning and problem-solving patterns. "
            "Focus on broadly applicable reasoning techniques, NOT domain-specific scenarios.\n"
            "Good examples: counterfactual reasoning, abductive reasoning, analogical reasoning, "
            "proof by contradiction, divide and conquer, causal inference, Bayesian updating.\n"
            "Bad examples: 'end-of-life ethical dilemma reasoning', 'epidemiological study design' "
            "(these are scenarios, not reasoning patterns).\n\n"
            "Organize by reasoning type (deductive, inductive, abductive, analogical, heuristic, "
            "probabilistic, etc.) and complexity level."
        ),
        generation_prompt_template=(
            "Generate {k} unique, general-purpose reasoning patterns that fit the category: {leaf_path}\n\n"
            "Each pattern must describe a general cognitive or analytical approach that can be applied "
            "across many domains. Do NOT tie patterns to specific domains or scenarios.\n\n"
            "Previously generated patterns for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): General pattern name (e.g. 'Counterfactual Reasoning', not 'Epidemiological Counterfactual Analysis')\n"
            "- category (string): Category of reasoning\n"
            "- description (string): How this pattern works in general (2-3 sentences)\n"
            "- when_to_use (string): When this pattern is most effective (1-2 sentences)"
        ),
        specificity_guidance="Each pattern should be a general reasoning technique applicable across many domains, not tied to a specific scenario.",
    ),
    "emotional_state": CategoryConfig(
        name="emotional_state",
        display_name="Emotional States",
        fields=[
            FieldSpec("category", "string", "Broad emotion category (e.g. 'Hopefulness', 'Frustration', 'Grief', 'Excitement')"),
            FieldSpec("name", "string", "Specific name for this emotional state"),
            FieldSpec("intensity", "string", "Intensity level (subtle/mild/moderate/strong/overwhelming)"),
            FieldSpec("valence", "string", "Emotional valence (positive/negative/mixed/neutral)"),
            FieldSpec("behavioral_description", "string", "How this state manifests in behavior (2-3 sentences)"),
            FieldSpec("example", "string", "An example situation or scenario where this emotional state commonly occurs"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of emotional states organized by valence "
            "(positive/negative/mixed), arousal level (calm to intense), and "
            "social context (solitary, interpersonal, group). Each leaf should "
            "be a very specific emotional state (e.g. 'the bittersweet nostalgia "
            "of revisiting a childhood home that has been renovated beyond "
            "recognition' not just 'sad')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific emotional states that fit the category: {leaf_path}\n\n"
            "Each state must be a nuanced, specific emotional experience with behavioral details. "
            "Avoid generic emotion labels.\n\n"
            "Previously generated states for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- category (string): Broad emotion category (e.g. 'Hopefulness', 'Frustration', 'Grief') - "
            "a simple, high-level emotion family\n"
            "- name (string): Specific name for this emotional state\n"
            "- intensity (string): one of subtle/mild/moderate/strong/overwhelming\n"
            "- valence (string): one of positive/negative/mixed/neutral\n"
            "- behavioral_description (string): How this state manifests in behavior (2-3 sentences)\n"
            "- example (string): A concrete example situation or scenario where this emotional state commonly occurs"
        ),
        specificity_guidance="Each state should be a nuanced emotional experience, not a basic emotion label.",
    ),
    "instruction_complexity": CategoryConfig(
        name="instruction_complexity",
        display_name="Instruction Complexity",
        fields=[
            FieldSpec("name", "string", "Short name for this instruction complexity type (e.g. 'Multi-step with implicit constraints', 'Contradictory requirements')"),
            FieldSpec("level", "string", "Complexity level (simple/moderate/complex/expert/ambiguous)"),
            FieldSpec("ambiguity", "string", "Ambiguity level (none/low/moderate/high/contradictory)"),
            FieldSpec("description", "string", "What makes this type of instruction complex (2-3 sentences). Must be generic and not tied to any specific domain or scenario."),
            FieldSpec("example", "string", "A concrete example instruction at this complexity level"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of instruction complexity types organized by the "
            "NATURE of the complexity, NOT by domain.\n\n"
            "Good complexity dimensions:\n"
            "- Structural: multi-step, nested conditions, ordering dependencies\n"
            "- Ambiguity: vague requirements, implicit assumptions, contradictions\n"
            "- Constraint: conflicting constraints, hidden constraints, optimization trade-offs\n"
            "- Scope: underspecified scope, shifting requirements, meta-instructions\n"
            "- Communication: indirect requests, implied context, cultural assumptions\n\n"
            "Each leaf should describe a TYPE of instruction complexity, not a specific scenario.\n"
            "Good: 'Multi-step instructions with implicit ordering constraints'\n"
            "Bad: 'QoS configuration for network interfaces' (this is a scenario, not a complexity type)"
        ),
        generation_prompt_template=(
            "Generate {k} unique instruction complexity types that fit the category: {leaf_path}\n\n"
            "Each entry must describe a GENERIC type of instruction complexity that could apply to "
            "ANY domain. The description should be about what makes this type of instruction complex, "
            "not about a specific task or scenario. The example can be domain-specific to illustrate.\n\n"
            "Previously generated examples for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): Short descriptive name for this complexity type\n"
            "- level (string): one of simple/moderate/complex/expert/ambiguous\n"
            "- ambiguity (string): one of none/low/moderate/high/contradictory\n"
            "- description (string): What makes this type of instruction complex (2-3 sentences, generic, not domain-specific)\n"
            "- example (string): A concrete example instruction at this complexity level"
        ),
        specificity_guidance="Each entry should describe a generic instruction complexity type applicable to any domain, not a domain-specific scenario.",
    ),
    "tool_group": CategoryConfig(
        name="tool_group",
        display_name="Tool Groups",
        fields=[
            FieldSpec("domain", "string", "Tool group domain (e.g. 'Flight Booking')"),
            FieldSpec("description", "string", "What this tool group does"),
            FieldSpec("taxonomy_path", "string", "Taxonomy path this group belongs to"),
            FieldSpec("tools_json", "string", "JSON-serialized list of ToolFunction dicts"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of LLM tool/function calling domains organized by "
            "service category and specialization. These represent groups of related "
            "tools that an AI agent would use together.\n\n"
            "Root categories:\n"
            "- Communication (email, messaging, calendar)\n"
            "- Data & Storage (databases, file systems)\n"
            "- Web & Search (scraping, search, browser automation)\n"
            "- Code & Development (git, CI/CD, code execution)\n"
            "- Finance & Commerce (payments, e-commerce, invoicing)\n"
            "- Travel & Logistics (flights, hotels, maps, shipping)\n"
            "- Media & Content (image, audio, video, documents)\n"
            "- Identity & Auth (OAuth, user management)\n"
            "- AI & ML (inference, embeddings, vector stores)\n"
            "- Monitoring & Ops (logging, metrics, alerting)\n"
            "- IoT & Hardware (sensors, device control)\n"
            "- Healthcare (patient records, appointments)\n\n"
            "Each leaf should be a specific tool domain "
            "(e.g. 'Flight search and booking with seat selection' not just 'travel')."
        ),
        generation_prompt_template="",  # Not used; tool_sampler has its own prompts
        specificity_guidance="Each tool group should represent a coherent set of related API functions with inter-tool dependencies.",
    ),
}
