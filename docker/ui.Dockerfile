FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY ui/package*.json ./
RUN npm install

# Copy source code
COPY ui/ ./

# Build the application
RUN npm run build

# Remove dev dependencies after build
RUN npm prune --production

EXPOSE 3000

CMD ["npm", "start"]
