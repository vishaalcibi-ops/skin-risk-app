# add transparency to style.css
css_append = """
body.auth-page-bg .app-container {
    background-color: transparent !important;
}

/* Also ensure no auth-specific wrapper blocks it */
.auth-container-wrapper {
    background-color: transparent !important;
}
"""

with open('static/style.css', 'a', encoding='utf-8') as f:
    f.write(css_append)

print("Transparency bug for auth page background fixed!")
