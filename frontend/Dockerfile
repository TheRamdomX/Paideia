# Dockerfile para el Frontend de Paideia
# Sirve archivos estáticos con Nginx

FROM nginx:alpine

# Copiar configuración personalizada de nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar archivos del frontend
COPY index.html /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY styles.css /usr/share/nginx/html/

# Exponer puerto 80
EXPOSE 80

# Nginx se ejecuta en foreground por defecto
CMD ["nginx", "-g", "daemon off;"]
