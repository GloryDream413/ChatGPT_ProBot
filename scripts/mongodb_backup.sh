#!/bin/bash

# add cron job  with "crontab -e" command
## 0 */1 * * * bash /home/ubuntu/chatgpt_telegram_bot_pro/scripts/mongodb_backup.sh

# set the directory for the backups
BACKUP_DIR="/home/ubuntu/mongodb_backup"

# create a directory based on the current date
DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_PATH="$BACKUP_DIR/$DATE"
mkdir -p $BACKUP_PATH

echo $BACKUP_PATH

# create the mongodump
mongodump --out $BACKUP_PATH

# # Remove old backups (older than 7 days)
find $BACKUP_DIR/* -mtime +7 -exec rm -rf {} \;
