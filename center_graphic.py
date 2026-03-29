css_to_append = """
@media (min-width: 769px) {
    .hero-graphic {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 100% !important;
        position: relative !important;
    }
}
"""

with open('static/style.css', 'a', encoding='utf-8') as f:
    f.write(css_to_append)
print("Hero-graphic centered via appended CSS!")
