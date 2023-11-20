<script>
  import '../app.css';
  import { onMount } from 'svelte';

  let readingPassageData = null;
  let recording = false;
  let audioBlob = null;

  let processingAudio = false
  let readingResultData = null;

  onMount(async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/passages/?level=3');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      let json = await response.json();
      readingPassageData = JSON.parse(json);
    } catch (error) {
      console.error('There has been a problem with your fetch operation:', error);
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };

      mediaRecorder.start();
      recording = true;

      mediaRecorder.onstop = () => {
        audioBlob = new Blob(audioChunks);
        recording = false;
      };

      setTimeout(() => {
        mediaRecorder.stop()
        setTimeout(() => {
          handleSubmit();
        }, 2000);
      }, 5000); // Stop recording after 5 seconds (adjust as needed)
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  }

  async function handleSubmit() {
    processingAudio = true;
    if (audioBlob) {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      // Submit the form data with the audio
      const response = await fetch('http://127.0.0.1:8000/upload-audio/?passage='+readingPassageData?.passage, {
        method: 'POST',
        body: formData,
      });

      // Handle the response
      let json = await response.json();
      readingResultData = JSON.parse(json);

      console.log("result {}", readingResultData);

      audioBlob = null;
      processingAudio = false;
      recording = false;
    }
  }
</script>

<div class="flex flex-col min-h-screen">
  <!-- Header -->
  <header class="bg-gray-800 text-white p-4">
    <h1>Reading with OpenAI</h1>
  </header>

  <!-- Main Content -->
  <main class="flex flex-1 overflow-auto">
    <!-- Left Column -->
    <div class="w-3/4 p-8 h-full">
      <!-- Rounded Pane with Header -->
      <div class="bg-white p-8 rounded-lg">
        {#if readingPassageData}
        <h2 class="text-xl font-bold mb-4">Let's try reading this.</h2>
        {/if}
        <!-- Quoted Reading in a Rounded Pane -->
        <div class="quote-pane bg-gray-100 p-6 rounded-lg">
          <form on:submit={handleSubmit}>

          {#if readingPassageData}
          <p class="text-lg">"{readingPassageData.passage}"</p>

          {#if !recording && !audioBlob}
            <button class="mt-4 px-4 py-2 bg-blue-500 text-white rounded" on:click={startRecording}>Start Reading</button>
          {:else if !audioBlob}
            <p class="mt-4 text-green-500">Recording...</p>
          {:else}
            <p class="mt-4 text-blue-500">Analyzing your reading...</p>
          {/if}
          {:else}
          <p>Loading data...</p>
          {/if}
          </form>
        </div>
      </div>

      <!-- Reading result area -->
      {#if readingResultData}
      <div class="bg-white p-8 mt-8 rounded-lg">
        <h2 class="text-xl font-bold mb-4">Here's the results.</h2>

        <!-- This is the analysis of the reading... -->
        <div class="quote-pane bg-gray-100 p-6 rounded-lg">
          <p class="text-lg font-bold">Original:</p>
          "{readingResultData.passage}"
          <p class="text-lg font-bold">Your Reading:</p>
          "{readingResultData.reading}"
          <p class="text-lg font-bold">Suggestions:</p>
          <ul>
          {#each readingResultData?.suggestions || [] as {area, suggestion}}
            <li>{area}: {suggestion}</li>
          {/each}
          </ul>
        </div>
      </div>
      {/if}

    </div>

    <!-- Right Column -->
    <div class="w-1/4 bg-gray-200 p-4">
      <!-- Right column content -->
      {#if readingPassageData}
        <div>
          <h3 class="text-lg font-semibold">Author</h3>
          <p>{readingPassageData.author}</p>
        </div>
        <div>
          <h3 class="text-lg font-semibold">Source</h3>
          <p>{readingPassageData.source}</p>
        </div>
        <div>
          <h3 class="text-lg font-semibold">Reason for Selection</h3>
          <p>{readingPassageData.reason}</p>
        </div>
      {/if}
    </div>
  </main>

  <!-- Footer -->
  <footer class="bg-gray-800 text-white p-4">
    <p>Footer Content</p>
  </footer>
</div>

<style lang="postcss">
  :global(html) {
    background-color: theme(colors.gray.100);
  }
  .quote-pane {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
</style>
