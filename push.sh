
#!/bin/bash

# Добавить "o" в начало README.md
sed -i '' '1s/^/o/' lol.txt

git add .
git commit -m "1231"
git push -u origin main --force
