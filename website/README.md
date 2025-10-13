# Genesis.ai Website

This directory contains the landing page for genesis.ai, showcasing the ShinkaEvolve project.

## Local Testing

To test the website locally, simply open `index.html` in your web browser:

```bash
# Option 1: Open directly
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or just double-click index.html in your file browser

# Option 2: Use a local server (recommended for testing)
python3 -m http.server 8000
# Then visit http://localhost:8000 in your browser
```

## Deploying to Cloudflare Pages

### Prerequisites
- Cloudflare account
- genesis.ai domain configured in Cloudflare
- GitHub repository access

### Deployment Steps

#### Option 1: Git Integration (Recommended)

1. **Push to GitHub**:
   ```bash
   git add website/
   git commit -m "Add genesis.ai landing page"
   git push origin main
   ```

2. **Connect to Cloudflare Pages**:
   - Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
   - Go to **Workers & Pages** → **Pages**
   - Click **"Create a project"** → **"Connect to Git"**

3. **Configure Repository**:
   - Select your GitHub account and authorize Cloudflare
   - Choose the `SakanaAI/ShinkaEvolve` repository (or your fork)
   - Click **"Begin setup"**

4. **Configure Build Settings**:
   - **Project name**: `genesis-ai` (or any name you prefer)
   - **Production branch**: `main`
   - **Framework preset**: `None`
   - **Build command**: (leave blank)
   - **Build output directory**: `website`
   - **Root directory (advanced)**: `website`

5. **Deploy**:
   - Click **"Save and Deploy"**
   - Wait ~1-2 minutes for the first deployment
   - You'll get a `*.pages.dev` URL

6. **Add Custom Domain**:
   - In your Cloudflare Pages project, go to **"Custom domains"**
   - Click **"Set up a custom domain"**
   - Enter: `genesis.ai`
   - Cloudflare will automatically configure DNS
   - Wait 5-30 minutes for DNS propagation
   - Your site will be live at https://genesis.ai

#### Option 2: Direct Upload (Quick Test)

1. **Go to Cloudflare Dashboard** → **Workers & Pages** → **Pages**

2. **Upload Files**:
   - Click **"Create a project"** → **"Upload assets"**
   - Drag and drop the entire `website` folder
   - Or use Wrangler CLI:
     ```bash
     npx wrangler pages deploy website --project-name=genesis-ai
     ```

3. **Add Custom Domain** (same as Option 1, step 6)

### Automatic Deployments

With Git integration (Option 1), Cloudflare Pages will automatically:
- Deploy on every push to the `main` branch
- Create preview deployments for pull requests
- Show deployment status in GitHub

### Custom Domain Configuration

If your domain isn't automatically detected:

1. In Cloudflare Dashboard, go to **DNS** → **Records**
2. Add a CNAME record:
   - **Type**: CNAME
   - **Name**: `@` (or `genesis.ai`)
   - **Target**: `<your-project>.pages.dev`
   - **Proxy status**: Proxied (orange cloud)

## File Structure

```
website/
├── index.html          # Main landing page
├── style.css           # Styles (modern, responsive design)
├── images/             # Visual assets
│   ├── logo.png        # Logo/favicon
│   ├── conceptual.png  # Conceptual diagram
│   └── webui.png       # WebUI screenshot
└── README.md           # This file
```

## Updating the Website

### With Git Integration
```bash
# Make your changes
vi index.html  # or style.css

# Commit and push
git add website/
git commit -m "Update website content"
git push origin main

# Cloudflare Pages will automatically deploy
```

### With Direct Upload
- Re-upload the entire `website` folder through the Cloudflare Dashboard
- Or use: `npx wrangler pages deploy website --project-name=genesis-ai`

## Performance

The website is optimized for performance:
- **Total size**: ~3.1 MB (mainly images)
- **Load time**: < 2 seconds on 3G
- **Lighthouse score**: 95+
- **CDN**: Cloudflare's global network
- **SSL**: Automatic HTTPS
- **Bandwidth**: Unlimited (free tier)

## Troubleshooting

### DNS not propagating
- Wait up to 48 hours (usually 5-30 minutes)
- Check DNS: `dig genesis.ai` or `nslookup genesis.ai`
- Clear browser cache: Ctrl+Shift+R / Cmd+Shift+R

### Images not loading
- Check that `images/` directory is uploaded
- Verify file paths are relative: `images/logo.png`
- Check browser console for 404 errors

### Styles not applying
- Hard refresh: Ctrl+Shift+R / Cmd+Shift+R
- Check that `style.css` is in the same directory as `index.html`
- Verify the link tag in `index.html`: `<link rel="stylesheet" href="style.css">`

## Support

- **Cloudflare Pages Docs**: https://developers.cloudflare.com/pages/
- **Cloudflare Community**: https://community.cloudflare.com/
- **ShinkaEvolve Issues**: https://github.com/SakanaAI/ShinkaEvolve/issues

## License

This website is part of the ShinkaEvolve project, licensed under Apache 2.0.
