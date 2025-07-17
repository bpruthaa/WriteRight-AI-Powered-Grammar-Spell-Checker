import os
import re
import streamlit as st
from spellchecker import SpellChecker
import language_tool_python
from transformers import pipeline

# Suppress TensorFlow warnings from transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Initialize spellchecker
spell = SpellChecker()

# Initialize LanguageTool (UK English) for grammar checking
tool = language_tool_python.LanguageTool('en-GB')

@st.cache_resource
def load_grammar_model():
    return pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction",
        device=-1  # CPU only
    )

grammar_corrector = load_grammar_model()

def correct_grammar(text):
    result = grammar_corrector(text, max_length=128, clean_up_tokenization_spaces=True)
    return result[0]['generated_text']

def find_spelling_issues(text):
    words = re.findall(r'\b\w+\b', text)
    issues = []
    idx = 0
    for word in words:
        start = text.find(word, idx)
        end = start + len(word)
        idx = end
        if word.lower() not in spell:
            suggestions = list(spell.candidates(word))
            if suggestions:
                suggestions = sorted(suggestions, key=lambda w: spell.distance(word, w))
            issues.append({
                "offset": start,
                "errorLength": len(word),
                "replacements": suggestions[:5] if suggestions else [],
                "message": "Possible spelling mistake",
                "type": "spelling"
            })
    return issues

def highlight_grammar_issues(text):
    return tool.check(text)

def highlight_text_with_issues(text, issues):
    issues = sorted(issues, key=lambda x: x['offset'])
    offset_correction = 0
    for issue in issues:
        start = issue['offset'] + offset_correction
        end = start + issue['errorLength']
        color = {
            "spelling": "red",
            "grammar": "blue",
            "capitalization": "orange"
        }.get(issue['type'], "black")

        suggestions = ', '.join(issue['replacements'][:5]) if issue['replacements'] else "No suggestions"
        title = f"Suggestions: {suggestions}\nIssue: {issue['message']}"

        highlighted = (
            f"<span style='text-decoration: underline; "
            f"text-decoration-color: {color}; cursor: help;' title='{title}'>"
            f"{text[start:end]}</span>"
        )

        text = text[:start] + highlighted + text[end:]
        offset_correction += len(highlighted) - (end - start)
    return text

# Streamlit UI

st.set_page_config(page_title="WriteRight - Spell & Grammar Checker", page_icon="✍️")
st.title("✍️ WriteRight: Spell & Grammar Checker")

user_input = st.text_area("📝 Enter your text here:", height=150)

if st.button("Check Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        st.subheader("🧠 Grammar Correction:")
        corrected_text = correct_grammar(user_input)
        st.write(corrected_text)

        st.subheader("🛠 Issues Highlighted (Original Text):")

        spelling_issues = find_spelling_issues(user_input)
        grammar_issues_raw = highlight_grammar_issues(user_input)
        grammar_issues = [{
            "offset": issue.offset,
            "errorLength": issue.errorLength,
            "replacements": issue.replacements,
            "message": issue.message,
            "type": "grammar"
        } for issue in grammar_issues_raw]

        all_issues = spelling_issues + grammar_issues

        if not all_issues:
            st.success("🎉 No issues found!")
        else:
            highlighted_text = highlight_text_with_issues(user_input, all_issues)
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.markdown("---")
            st.write("**Detected Issues:**")
            for issue in all_issues:
                st.markdown(f"❌ **Issue:** {issue['message']}")
                st.markdown(f"🔁 **Suggestion(s):** {', '.join(issue['replacements']) if issue['replacements'] else 'None'}")
                snippet = user_input[issue['offset']:issue['offset'] + issue['errorLength']]
                st.markdown(f"📍 **At:** `{snippet}`")
                st.markdown("---")

        st.subheader("📘 Final Corrected Version:")
        st.text_area("Corrected Text", corrected_text, height=150)
