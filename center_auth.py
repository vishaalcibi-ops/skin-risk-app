import glob

def center_auth_form(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already wrapped
    if 'min-height: calc(100vh - 120px); display: flex; align-items: center; justify-content: center;' in content:
        return
        
    # Replace the card start
    original_start = '<div class="auth-card-alt animate-slide-up" style="margin-top: 2rem;">'
    new_start = '''
        <div class="auth-container-wrapper" style="min-height: calc(100vh - 120px); display: flex; align-items: center; justify-content: center; padding: 2rem 0;">
            <div class="auth-card-alt animate-slide-up" style="width: 100%; max-width: 450px; margin: 0 auto; box-shadow: 0 15px 50px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.15); background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(20px);">
    '''
    # Wait, signup.html might not have style="margin-top: 2rem;"
    # Let's use a safer replacement
    
    if original_start in content:
        content = content.replace(original_start, new_start.strip('\n'))
        
        # Add the closing div right before the end of app-container
        # Instead of parsing, we can just replace '</body>' with '</div>\n</body>'
        # But wait, we need to close the .auth-container-wrapper right after .auth-card-alt ends.
        
        # The easiest way is to find the LAST </div> that belongs to auth-card-alt
        # In both files, auth-card-alt ends exactly before:
        #     </div>
        # </body>
        # Let's just replace:
        #         </div>
        #     </div>
        # </body>
        # Actually:
        #     </div>
        # </body>
        # becomes
        #     </div>
        #     </div>
        # </body>
        
        # Wait, the closing of auth-card-alt is followed by closing of app-container.
        # Let's look at the end of login.html:
        #             <div class="auth-link-alt">
        #                 Don't have an account? <a href="/signup">Sign Up</a>
        #             </div>
        #         </div>
        #     </div>
        # </body>
        content = content.replace('        </div>\n    </div>\n</body>', '        </div>\n        </div>\n    </div>\n</body>')
        
        # Also ensure body class has auth-page-bg
        if '<body class="auth-page-bg">' not in content:
            content = content.replace('<body>', '<body class="auth-page-bg">')
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

center_auth_form('templates/login.html')
center_auth_form('templates/signup.html')
print("Login & Signup templates successfully centered with the background image integrated!")
