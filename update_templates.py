import os

template_dir = 'templates'
files = os.listdir(template_dir)
nav_str = '''        <!-- Desktop Navbar -->
        <header class="navbar desktop-only">
            <div class="logo">
                <i class="fa-solid fa-microscope text-gradient"></i>
                <span>Skin_<span class="text-gradient">Risk</span></span>
            </div>
            <nav class="nav-links">
                {% if current_user and current_user.is_authenticated %}
                <span class="user-welcome"><i class="fa-solid fa-circle-user"></i> Hi, {{ current_user.name or
                    current_user.email.split('@')[0] }}</span>
                <a href="/dashboard"><i class="fa-solid fa-house"></i> Home</a>
                <a href="/history"><i class="fa-solid fa-clock-rotate-left"></i> History</a>
                <a href="/diary"><i class="fa-solid fa-book-medical"></i> Diary</a>
                <a href="/routine"><i class="fa-solid fa-sparkles"></i> Routine</a>
                <a href="/profile"><i class="fa-solid fa-user"></i> Profile</a>
                <a href="/logout" class="logout-link"><i class="fa-solid fa-right-from-bracket"></i> Logout</a>
                {% else %}
                <a href="/login"><i class="fa-solid fa-right-to-bracket"></i> Login</a>
                <a href="/signup"><i class="fa-solid fa-user-plus"></i> Signup</a>
                {% endif %}
            </nav>
        </header>

        <!-- Android Top Bar -->
        <header class="top-bar mobile-only">'''

for file in files:
    if file.endswith('.html') and file not in ['index.html', 'signup.html', 'login.html']:
        filepath = os.path.join(template_dir, file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        updated = False
        
        # Add desktop navbar and modify top bar
        if '<header class="top-bar">' in content:
            content = content.replace('<header class="top-bar">', nav_str)
            updated = True
        elif '<header class="top-bar' in content and 'mobile-only' not in content:
            # Handle top-bar with extra classes
            header_end_idx = content.find('>', content.find('<header class="top-bar'))
            header_full = content[content.find('<header class="top-bar'):header_end_idx+1]
            new_header = header_full.replace('class="top-bar', 'class="top-bar mobile-only')
            content = content.replace(header_full, nav_str.replace('<header class="top-bar mobile-only">', '') + new_header)
            updated = True
            
        # Modify bottom nav
        if '<nav class="bottom-nav">' in content:
            content = content.replace('<nav class="bottom-nav">', '<nav class="bottom-nav mobile-only">')
            updated = True
            
        if updated:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {file}")

print("Template updates complete.")
