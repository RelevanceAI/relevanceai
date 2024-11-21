from firecrawl import FirecrawlApp

app = FirecrawlApp(api_key='fc-cc05418ab0c04007b5aba4b7b5f5d0b9')

map_result = app.map_url('https://mendable.ai', params={
	'includeSubdomains': True
})