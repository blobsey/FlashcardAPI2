#!/bin/bash

# Script does a few things: 
# - Nginx and Certbot
# - Fetches SSL certs via Certbot
# - Sets up Nginx on port 8000 (default Uvicorn port)
# - Installs systemd timer to renew certbot certs

# Define the hostname and email variables with default values
HOSTNAME="example.com" # The base API URL
EMAIL="email@example.com" # Email used for Let's Encrypt notifications


# Make sure HOSTNAME and EMAIL are set to non-default values
if [ "$HOSTNAME" == "example.com" ]; then
    echo "Error: HOSTNAME is set to the default value. Please set the HOSTNAME variable to your actual domain."
    exit 1
fi

if [ "$EMAIL" == "your-email@example.com" ]; then
    echo "Error: EMAIL is set to the default value. Please set the EMAIL variable to your actual email address."
    exit 1
fi

# Install Nginx & Certbot
sudo dnf install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
sudo dnf install -y certbot python3-certbot-nginx

# Create an initial Nginx configuration for domain verification
NGINX_CONF="/etc/nginx/conf.d/fastapi.conf"
cat << EOF | sudo tee $NGINX_CONF
server {
    listen 80;
    server_name $HOSTNAME;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }
}
EOF

# Create directory for Let's Encrypt verification
sudo mkdir -p /var/www/letsencrypt

# Test Nginx configuration
sudo nginx -t

# Reload Nginx to apply the new configuration
if [ $? -eq 0 ]; then
    sudo systemctl reload nginx
    echo "Nginx has been reloaded with the new configuration."
else
    echo "Nginx configuration test failed. Please check the configuration file for errors."
    exit 1
fi

# Obtain SSL certificates from Let's Encrypt
sudo certbot certonly --webroot -w /var/www/letsencrypt -d $HOSTNAME --email $EMAIL --agree-tos --non-interactive

# Modify Nginx configuration for HTTPS
cat << EOF | sudo tee $NGINX_CONF
server {
    listen 80;
    server_name $HOSTNAME;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name $HOSTNAME;

    ssl_certificate /etc/letsencrypt/live/$HOSTNAME/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$HOSTNAME/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Test new Nginx configuration
sudo nginx -t

# Reload Nginx to apply HTTPS configuration
if [ $? -eq 0 ]; then
    sudo systemctl reload nginx
    echo "Nginx has been reloaded with the new HTTPS configuration."
else
    echo "Nginx configuration test failed. Please check the configuration file for errors."
    exit 1
fi

# Set up systemd timer for automatic certificate renewal

# Create a systemd service for certbot renewal
sudo bash -c 'cat << EOF > /etc/systemd/system/certbot-renew.service
[Unit]
Description=Certbot Renewal
Documentation=https://certbot.eff.org/

[Service]
ExecStart=/usr/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF'

# Create a systemd timer for certbot renewal
sudo bash -c 'cat << EOF > /etc/systemd/system/certbot-renew.timer
[Unit]
Description=Timer to renew certbot certificates

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF'

# Reload systemd to apply the new service and timer
sudo systemctl daemon-reload

# Enable and start the certbot-renew timer
sudo systemctl enable certbot-renew.timer
sudo systemctl start certbot-renew.timer

echo "Nginx is configured to serve HTTPS and a systemd timer has been set up for automatic certificate renewal."
