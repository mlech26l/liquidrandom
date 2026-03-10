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
            FieldSpec("title", "string", "Specific job title"),
            FieldSpec("industry", "string", "Industry sector"),
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
            "- title (string): Specific job title\n"
            "- industry (string): Industry sector\n"
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
            FieldSpec("description", "string", "Detailed task description (2-4 sentences)"),
            FieldSpec("constraints", "array of strings", "2-4 specific constraints"),
            FieldSpec("expected_behavior", "string", "What the solution should do"),
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
            "Each task must be a concrete programming challenge with clear requirements. "
            "Avoid vague or overly broad tasks.\n\n"
            "Previously generated tasks for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- title (string): Short, descriptive title\n"
            "- language (string): Programming language\n"
            "- difficulty (string): one of easy/medium/hard/expert\n"
            "- description (string): Detailed task description (2-4 sentences)\n"
            "- constraints (array of strings): 2-4 specific constraints\n"
            "- expected_behavior (string): What the solution should do"
        ),
        specificity_guidance="Each task should specify exact algorithms, data structures, or techniques needed.",
    ),
    "math_category": CategoryConfig(
        name="math_category",
        display_name="Math Categories",
        fields=[
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
            FieldSpec("name", "string", "Name of the writing style"),
            FieldSpec("tone", "string", "Overall tone"),
            FieldSpec("characteristics", "array of strings", "3-5 stylistic characteristics"),
            FieldSpec("description", "string", "Description of the style (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of writing styles organized by register "
            "(formal/informal), purpose (persuasive, informative, creative, etc.), "
            "and cultural/historical context. Each leaf should be a very specific "
            "style (e.g. 'Victorian epistolary prose with satirical undertones' "
            "not just 'formal writing')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific writing styles that fit the category: {leaf_path}\n\n"
            "Each style must be distinctive and well-characterized. "
            "Avoid generic or overlapping styles.\n\n"
            "Previously generated styles for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): Distinctive name for the style\n"
            "- tone (string): Overall tone\n"
            "- characteristics (array of strings): 3-5 stylistic characteristics\n"
            "- description (string): Description of the style (2-3 sentences)"
        ),
        specificity_guidance="Each style should be distinctive enough that two different writers would produce noticeably different text.",
    ),
    "scenario": CategoryConfig(
        name="scenario",
        display_name="Scenarios",
        fields=[
            FieldSpec("title", "string", "Short title"),
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
            "- title (string): Short, descriptive title\n"
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
            FieldSpec("name", "string", "Domain name"),
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
            FieldSpec("name", "string", "Topic name"),
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
            FieldSpec("name", "string", "Language name"),
            FieldSpec("region", "string", "Geographic region"),
            FieldSpec("register", "string", "Register (formal/informal/colloquial/technical/literary)"),
            FieldSpec("script", "string", "Writing script used"),
            FieldSpec("cultural_notes", "string", "Cultural context and usage notes (2-3 sentences)"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of languages and linguistic varieties organized "
            "by language family, geographic region, and sociolinguistic register. "
            "Each leaf should be a very specific linguistic variety "
            "(e.g. 'Shanghainese Wu Chinese as spoken by elderly residents in "
            "the Old City district' not just 'Chinese')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific language varieties that fit the category: {leaf_path}\n\n"
            "Each entry must describe a specific linguistic variety with cultural context. "
            "Avoid generic language entries.\n\n"
            "Previously generated entries for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): Language/variety name\n"
            "- region (string): Geographic region\n"
            "- register (string): Register (formal/informal/colloquial/technical/literary)\n"
            "- script (string): Writing script used\n"
            "- cultural_notes (string): Cultural context and usage notes (2-3 sentences)"
        ),
        specificity_guidance="Each entry should describe a specific linguistic variety with its social context, not just a language name.",
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
            "Generate a taxonomy of reasoning and problem-solving patterns "
            "organized by type (deductive, inductive, abductive, analogical, "
            "heuristic, etc.), domain of application, and complexity level. "
            "Each leaf should be a very specific reasoning approach "
            "(e.g. 'counterfactual reasoning applied to causal inference in "
            "epidemiological studies' not just 'logical reasoning')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific reasoning patterns that fit the category: {leaf_path}\n\n"
            "Each pattern must describe a specific cognitive or analytical approach. "
            "Avoid generic reasoning categories.\n\n"
            "Previously generated patterns for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- name (string): Specific pattern name\n"
            "- category (string): Category of reasoning\n"
            "- description (string): How this pattern works (2-3 sentences)\n"
            "- when_to_use (string): When this pattern is most effective (1-2 sentences)"
        ),
        specificity_guidance="Each pattern should describe a specific reasoning technique with clear application context.",
    ),
    "emotional_state": CategoryConfig(
        name="emotional_state",
        display_name="Emotional States",
        fields=[
            FieldSpec("name", "string", "Name of the emotional state"),
            FieldSpec("intensity", "string", "Intensity level (subtle/mild/moderate/strong/overwhelming)"),
            FieldSpec("valence", "string", "Emotional valence (positive/negative/mixed/neutral)"),
            FieldSpec("behavioral_description", "string", "How this state manifests in behavior (2-3 sentences)"),
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
            "- name (string): Specific name for this emotional state\n"
            "- intensity (string): one of subtle/mild/moderate/strong/overwhelming\n"
            "- valence (string): one of positive/negative/mixed/neutral\n"
            "- behavioral_description (string): How this state manifests in behavior (2-3 sentences)"
        ),
        specificity_guidance="Each state should be a nuanced emotional experience, not a basic emotion label.",
    ),
    "instruction_complexity": CategoryConfig(
        name="instruction_complexity",
        display_name="Instruction Complexity",
        fields=[
            FieldSpec("level", "string", "Complexity level (simple/moderate/complex/expert/ambiguous)"),
            FieldSpec("ambiguity", "string", "Ambiguity level (none/low/moderate/high/contradictory)"),
            FieldSpec("description", "string", "What makes this instruction complex (2-3 sentences)"),
            FieldSpec("example", "string", "A concrete example instruction at this complexity level"),
        ],
        taxonomy_seed_prompt=(
            "Generate a taxonomy of instruction types organized by complexity "
            "dimension (structural, conceptual, ambiguity, constraint density), "
            "domain, and specific challenge type. Each leaf should be a very "
            "specific type of instruction (e.g. 'multi-step data pipeline "
            "specification with implicit ordering constraints and unstated "
            "error handling requirements' not just 'complex instruction')."
        ),
        generation_prompt_template=(
            "Generate {k} unique, highly specific instruction complexity examples that fit the category: {leaf_path}\n\n"
            "Each entry must demonstrate a specific type of instruction complexity. "
            "Avoid generic complexity descriptions.\n\n"
            "Previously generated examples for this category (DO NOT repeat these):\n"
            "{existing_samples}\n\n"
            "Return a JSON object with a single key \"samples\" containing an array of {k} objects. "
            "Each object must have these exact fields:\n"
            "- level (string): one of simple/moderate/complex/expert/ambiguous\n"
            "- ambiguity (string): one of none/low/moderate/high/contradictory\n"
            "- description (string): What makes this instruction complex (2-3 sentences)\n"
            "- example (string): A concrete example instruction at this complexity level"
        ),
        specificity_guidance="Each entry should show a specific, realistic instruction that demonstrates the complexity type.",
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
