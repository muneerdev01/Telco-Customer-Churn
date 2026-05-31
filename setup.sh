#!/bin/bash

# Create Streamlit config directory
mkdir -p ~/.streamlit/

# Create Streamlit config file
cat > ~/.streamlit/config.toml << EOF
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = \$PORT
enableCORS = false
enableXsrfProtection = true

[client]
showErrorDetails = false
toolbarMode = "viewer"
EOF

echo "Streamlit configuration created successfully!"
