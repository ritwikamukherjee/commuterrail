# Modified from
# https://github.com/MaartenGr/streamlit_guide/blob/master/Procfile
# and
# https://github.com/MaartenGr/streamlit_guide/blob/master/setup.sh

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml