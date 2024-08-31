document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('summarize-form');
    const summaryDiv = document.getElementById('summary');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const text = document.getElementById('text').value;
        summaryDiv.innerHTML = '';
        loadingDiv.style.display = 'block';

        const response = await fetch(`/summarize?text=${encodeURIComponent(text)}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.he) {
                        const bulletPoint = document.createElement('li');
                        bulletPoint.textContent = data.he;
                        summaryDiv.appendChild(bulletPoint);
                    }
                }
            }
        }

        loadingDiv.style.display = 'none';
    });
});