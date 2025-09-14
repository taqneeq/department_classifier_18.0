# config/traits.py

from typing import List, Dict

TECHNICAL_TRAITS = [
    "coding_aptitude",
    "hardware_technical",
    "digital_design",
    "web_development"
]

CREATIVE_TRAITS = [
    "visual_creativity",
    "content_creation",
    "hands_on_crafting",
    "innovation_ideation"
]

PEOPLE_TRAITS = [
    "stakeholder_management",
    "team_collaboration",
    "public_interaction",
    "networking_ability"
]

ORGANIZATIONAL_TRAITS = [
    "logistics_coordination",
    "process_management",
    "event_execution",
    "strategic_planning"
]

BUSINESS_TRAITS = [
    "business_development",
    "financial_management",
    "leadership_initiative",
    "analytical_thinking"
]

ALL_TRAITS = (
    TECHNICAL_TRAITS +
    CREATIVE_TRAITS +
    PEOPLE_TRAITS +
    ORGANIZATIONAL_TRAITS +
    BUSINESS_TRAITS
)

TRAIT_DISPLAY_NAMES = {
    "coding_aptitude": "Coding Aptitude",
    "hardware_technical": "Hardware Technical",
    "digital_design": "Digital Design",
    "web_development": "Web Development",

    "visual_creativity": "Visual Creativity",
    "content_creation": "Content Creation",
    "hands_on_crafting": "Hands-on Crafting",
    "innovation_ideation": "Innovation & Ideation",

    "stakeholder_management": "Stakeholder Management",
    "team_collaboration": "Team Collaboration",
    "public_interaction": "Public Interaction",
    "networking_ability": "Networking Ability",

    "logistics_coordination": "Logistics Coordination",
    "process_management": "Process Management",
    "event_execution": "Event Execution",
    "strategic_planning": "Strategic Planning",

    "business_development": "Business Development",
    "financial_management": "Financial Management",
    "leadership_initiative": "Leadership Initiative",
    "analytical_thinking": "Analytical Thinking"
}

TRAIT_DESCRIPTIONS = {
    "coding_aptitude": "Programming, Arduino, Python skills",
    "hardware_technical": "Electronics, robotics, technical building",
    "digital_design": "Graphics, video editing, UI/UX",
    "web_development": "Frontend/backend, platforms, technical systems",

    "visual_creativity": "Design, aesthetics, artistic vision",
    "content_creation": "Writing, storytelling, messaging",
    "hands_on_crafting": "Physical art, installations, decor",
    "innovation_ideation": "Brainstorming, creative problem-solving",

    "stakeholder_management": "VIPs, guests, sponsors, authority figures",
    "team_collaboration": "Working within teams, peer coordination",
    "public_interaction": "Participants, audiences, general public",
    "networking_ability": "Building professional connections",

    "logistics_coordination": "Scheduling, venue management, supplies",
    "process_management": "Documentation, compliance, systematic work",
    "event_execution": "Day-of operations, real-time management",
    "strategic_planning": "Long-term vision, campaign development",

    "business_development": "Sponsorships, partnerships, deals",
    "financial_management": "Budgets, negotiations, cost optimization",
    "leadership_initiative": "Taking charge, decision-making, mentoring",
    "analytical_thinking": "Data analysis, metrics, performance tracking"
}

TRAIT_CATEGORIES = {
    **{trait: "Technical" for trait in TECHNICAL_TRAITS},
    **{trait: "Creative" for trait in CREATIVE_TRAITS},
    **{trait: "People & Communication" for trait in PEOPLE_TRAITS},
    **{trait: "Organizational" for trait in ORGANIZATIONAL_TRAITS},
    **{trait: "Business & Leadership" for trait in BUSINESS_TRAITS}
}
