import re

filepath = 'templates/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find start and end of the block to move
quiz_marker = '<!-- AI Symptom Checker Quiz -->'
end_marker = '<!-- HOW IT WORKS SECTION -->'
# We need to find the close of the HOW IT WORKS SECTION.
# It ends with:
#                     </div>
#                 </div>
#             </div>
#
#             <!-- Graphic side element -->

start_idx = content.find(quiz_marker)
end_idx = content.find('<!-- Graphic side element -->')

# The block to extract
block_to_move = content[start_idx:end_idx]

# Remove the block from its current location
content = content[:start_idx] + content[end_idx:]

# Find </main> and insert the block right after it
main_close_idx = content.find('</main>')

section_wrapper = f'''
        <section class="features-section" style="width: 100%; max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 2rem;">
            {block_to_move}
        </section>
'''

content = content[:main_close_idx + 7] + section_wrapper + content[main_close_idx + 7:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Move script completed successfully!")
